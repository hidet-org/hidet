raise ValueError()
from typing import Optional, Sequence, Union, List, Any, Tuple, ContextManager
from textwrap import indent
from collections import defaultdict
from typing import Type, Dict, Mapping
from hidet.ir.task import Task
from hidet.ir.func import IRModule
from hidet.ir.node import Node
from hidet.ir.dialects.pattern import TaskPattern, match
from hidet.utils import TableBuilder


class NotSupportedError(Exception):
    pass


class Schedule:
    def keys(self) -> List[Tuple[str, Union[int, float, str]]]:
        raise NotImplementedError()

    def derived_keys(self) -> List[Tuple[str, Union[int, float, str]]]:
        raise NotImplementedError()


class Implementer:
    def priority(self) -> int:
        raise NotImplementedError()

    def task_pattern(self) -> TaskPattern:
        raise NotImplementedError()

    def implement(self, task: Task, match: Mapping[Node, Any]) -> IRModule:
        raise NotImplementedError()

    def resolve(self, task, match, schedules: List[Schedule], ir_modules: List[IRModule], task_label: str, parallel=True, verbose=True) -> IRModule:
        # from hidet.runtime.value import dummy_inputs_from_task
        from hidet.backend import BuildInstance, batch_build
        import numpy as np
        assert len(schedules) == len(ir_modules)
        if len(ir_modules) == 1:
            return ir_modules[0]
        build_instances = [BuildInstance(ir_module=ir_module,
                                         output_dir=f'./outs/resolve/{task_label}/{idx}',
                                         keep_ir=False,
                                         nvcc_keep=False,
                                         verbose=False) for idx, ir_module in enumerate(ir_modules)]
        compiled_modules = batch_build(build_instances, parallel=parallel, verbose=verbose)
        dummy_inputs = dummy_inputs_from_task(task)
        best_latency = None
        best_ir_module = None
        latencies = []
        for schedule, ir_module, compiled_module in zip(schedules, ir_modules, compiled_modules):
            repeat_latency = compiled_module[task.name].profile(*dummy_inputs, warmup=2, number=1, repeat=10)
            latency = float(np.median(repeat_latency))
            latencies.append(latency)
            if best_latency is None or best_latency > latency:
                best_latency = latency
                best_ir_module = ir_module
        with TableBuilder(headers=['idx'] + [v[0] for v in (schedules[0].keys() + schedules[0].derived_keys())] + ['latency']) as tb:
            rows = []
            for idx, (schedule, latency) in enumerate(zip(schedules, latencies)):
                row = [idx] + [v[1] for v in schedule.keys() + schedule.derived_keys()] + [latency]
                rows.append(row)
            rows = sorted(rows, key=lambda v: v[-1])
            for row in rows:
                tb += row
        with open(f'./outs/resolve/{task_label}/report.txt', 'w') as f:
            f.write(str(tb))
        return best_ir_module

    @staticmethod
    def check(cond: bool, msg=""):
        if not cond:
            raise NotSupportedError(msg)


_impl_cls2name: Dict[Type[Implementer], str] = {}
_name2impl: Dict[str, Implementer] = {}
_name2priority: Dict[str, int] = {}
_priority2names: Dict[int, List[str]] = defaultdict(list)


class ImplementerContext:
    contexts: List['ImplementerContext'] = []

    def __init__(self, allowed: Optional[Sequence[Union[str, Type[Implementer]]]] = None, space_level=0):
        self.allowed: Optional[List[str]] = self._get_impl_names(allowed, None)  # None means all implementers
        self.space_level = space_level

    def __enter__(self):
        ImplementerContext.contexts.append(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        assert ImplementerContext.contexts[-1] is self
        ImplementerContext.contexts.pop()

    @classmethod
    def current(cls):
        return cls.contexts[-1]

    @staticmethod
    def _get_impl_names(impls: Optional[Sequence[Union[str, Type[Implementer]]]], default) -> List[str]:
        if impls is None:
            return default
        if not isinstance(impls, (tuple, list)):
            impls = [impls]
        return [impl if isinstance(impl, str) else _impl_cls2name[impl] for impl in impls]


ImplementerContext.contexts.append(ImplementerContext())  # fallback context, allow all implementers


def dummy_inputs_from_task(task: Task):
    from hidet.ir.type import TensorType
    from hidet.ir.expr import Constant
    from hidet.tos.tensor import randn
    inputs = []
    for idx, param_type in enumerate(task.param_types()):
        assert isinstance(param_type, TensorType)
        assert all(isinstance(s, Constant)for s in param_type.shape)
        stype = param_type.scalar_type.name
        scope = param_type.scope.name
        shape = [int(s) for s in param_type.shape]
        # strides = [int(s) for s in param_type.strides]
        scope2device = {
            'global': 'cuda',
            'host': 'cpu'
        }
        inputs.append(randn(shape, stype, device=scope2device[scope]))
    return inputs


def register_impl(name):
    if name in _name2impl:
        print(f"Implementer {name} has existed.")
        return lambda cls: cls

    def wrapper(cls: Type[Implementer]):
        impl = cls()
        _name2impl[name] = impl
        _impl_cls2name[cls] = name
        _name2priority[name] = impl.priority()
        _priority2names[impl.priority()].append(name)
        return cls

    return wrapper


def resolve_task(task: Task) -> Optional[Implementer]:
    implementers: List[List[str]] = []
    ctx = ImplementerContext.contexts[-1]
    added = set()
    if ctx.allowed is None:
        allowed = set(_name2impl.keys())
    else:
        allowed = set(ctx.allowed)

    priorities = sorted(_name2priority.values())
    for p in reversed(priorities):
        priority_impls = []
        for impl_name in _priority2names[p]:
            if impl_name not in added and impl_name in allowed:
                priority_impls.append(impl_name)
        if len(priority_impls) > 0:
            implementers.append(priority_impls)
            added.update(priority_impls)

    # try implementers groups by groups, the ones with higher priority try first
    for impls in implementers:
        for impl_name in impls:
            impl = _name2impl[impl_name]
            matching, msg = match(impl.task_pattern(), task)
            if matching:
                return impl
    return None


def implement(task: Task) -> IRModule:
    implementer = resolve_task(task)
    if implementer:
        matched, msg = match(implementer.task_pattern(), task)
        assert matched is not None
        return implementer.implement(task, matched)
    else:
        raise NotSupportedError('Can not implement task: \n{}'.format(str(task)))


def impl_context(allowed: Optional[Sequence[Union[str, Type[Implementer]]]] = None, space_level=0) -> ContextManager[ImplementerContext]:
    """
    Parameters
    ----------
    allowed: Optional[Sequence[Union[str, Type[Implementer]]]], default None
        Specify the allowed implementers. None means all implementers are allowed.
    space_level: int, default 0
        Search space level, candidates:
        0 - Use default schedule, with no search.
        1 - Search in a small group of schedules. The number of schedules should in range 1 to 32.
        2 - Exhaustive search in predefined schedule space. The number of schedules can be large (e.g., more than 100 schedules).
    Returns
    -------
    ret: ImplementerContext
        The implementer context. And be used with 'with' statement in python.
    """
    return ImplementerContext(allowed, space_level)
