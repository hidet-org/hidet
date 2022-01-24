from typing import Optional, Sequence, Union, List
from textwrap import indent
from collections import defaultdict
from typing import Type, Dict, Mapping
from hidet.ir.task import Task
from hidet.ir.func import IRModule
from hidet.ir.node import Node
from hidet.ir.dialects.pattern import TaskPattern, match


class NotSupportedError(Exception):
    pass


class Implementer:
    def priority(self) -> int:
        raise NotImplementedError()

    def task_pattern(self) -> TaskPattern:
        raise NotImplementedError()

    def implement(self, task: Task, match: Mapping[Node, Node]) -> IRModule:
        raise NotImplementedError()

    @staticmethod
    def check(cond: bool, msg=""):
        if not cond:
            raise NotSupportedError(msg)


# _implementers: Dict[int, Dict[str, Implementer]] = defaultdict(dict)
# _implementer2name: Dict[Type[Implementer], str] = {}
# _implementer_priorities: Dict[str, int] = {}
#
_impl_cls2name: Dict[Type[Implementer], str] = {}
_name2impl: Dict[str, Implementer] = {}
_name2priority: Dict[str, int] = {}
_priority2names: Dict[int, List[str]] = defaultdict(list)


class ImplementerContext:
    contexts: List['ImplementerContext'] = []

    def __init__(self,
                 disabled: Optional[Sequence[Union[str, Type[Implementer]]]] = None,
                 try_first: Optional[Sequence[Union[str, Type[Implementer]]]] = None):
        self.disabled: List[str] = [impl if isinstance(impl, str) else _impl_cls2name[impl] for impl in disabled] if disabled else []
        self.try_first: List[str] = [impl if isinstance(impl, str) else _impl_cls2name[impl] for impl in try_first] if try_first else []

    def __enter__(self):
        ImplementerContext.contexts.append(self)

    def __exit__(self, exc_type, exc_val, exc_tb):
        assert ImplementerContext.contexts[-1] is self
        ImplementerContext.contexts.pop()


# add a fall back context
ImplementerContext.contexts.append(ImplementerContext())


def register_impl(name):
    if name in _name2impl:
        raise KeyError("Implementer has existed.")

    def wrapper(cls: Type[Implementer]):
        impl = cls()
        _name2impl[name] = impl
        _impl_cls2name[cls] = name
        _name2priority[name] = impl.priority()
        _priority2names[impl.priority()].append(name)
        return cls

    return wrapper


def implement(task: Task) -> IRModule:
    implementers: List[List[str]] = []
    ctx = ImplementerContext.contexts[-1]
    added = set(ctx.disabled)

    # 1. try the 'try_first' implementers in order in current implementer context
    for impl_name in ImplementerContext.contexts[-1].try_first:
        if impl_name not in added:
            implementers.append([impl_name])
            added.add(impl_name)

    # 2. if failed, try all implementers except the 'disabled' ones in current context
    priorities = sorted(_name2priority.values())
    for p in reversed(priorities):
        priority_impls = []
        for impl_name in _priority2names[p]:
            if impl_name not in added:
                priority_impls.append(impl_name)
        if len(priority_impls) > 0:
            implementers.append(priority_impls)
            added.update(priority_impls)

    # try implementers groups by groups
    match_messages = {}
    impl_messages = {}
    ir_module = IRModule(funcs={}, task=task)
    for impls in implementers:
        for impl_name in impls:
            impl = _name2impl[impl_name]
            matching, msg = match(impl.task_pattern(), task)
            if matching:
                try:
                    impl_ir_module = impl.implement(task, matching)
                except NotSupportedError as e:
                    impl_messages[impl_name] = str(e)
                else:
                    if len(impl_ir_module.functions) == 0:
                        impl_messages[impl_name] = 'Empty ir module returned.'
                    else:
                        ir_module.include(impl_ir_module)
            else:
                impl_worker_cls = impl.task_pattern().worker.__class__
                task_worker_cls = task.worker.__class__
                if impl_worker_cls == task_worker_cls:
                    match_messages[impl_name] = msg
        if len(ir_module.functions) != 0:
            return ir_module

    match_report = "\n".join([f'{name}:\n{indent(msg, "    ")}' for name, msg in match_messages.items()])
    impl_report = "\n".join([f'{name}:\n{indent(msg, "    ")}' for name, msg in impl_messages.items()])
    lines = ['Can not implement task \'{}\''.format(task.name)]
    if len(match_messages) > 0:
        lines.append('Implementers failed to match: {}'.format(str(list(match_messages.keys()))))
    if len(impl_messages) > 0:
        lines.append('Implementers matched but failed to implement: {}'.format(str(list(impl_messages.keys()))))
    if len(match_messages) > 0:
        lines.append('Detail logs for each implementer failed to match: \n{}'.format(match_report))
    if len(impl_messages) > 0:
        lines.append('Detail logs for each implementer matched but failed to implement: \n{}'.format(impl_report))

    raise NotSupportedError('\n'.join(lines))


def impl_context(disabled: Optional[Sequence[Union[str, Type[Implementer]]]] = None,
                 try_first: Optional[Sequence[Union[str, Type[Implementer]]]] = None):
    return ImplementerContext(disabled, try_first)
