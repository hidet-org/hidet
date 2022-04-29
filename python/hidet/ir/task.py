from __future__ import annotations
import inspect
from typing import Dict, List, Union, Optional, Sequence, Type, Tuple, Callable
from hidet.ir.node import Node
from hidet.ir.expr import Expr, Var, TensorElement, var
from hidet.ir.func import IRModule
from hidet.ir.dialects.compute import TensorNode, ScalarNode


class Target:
    _supported_targets = ['cuda', 'cpu']

    def __init__(self, name: str, attrs: List[str]):
        if name not in self._supported_targets:
            raise ValueError('Does not support target {}, candidates {}.'.format(name, self._supported_targets))
        self.name = name
        self.attrs = attrs

    @staticmethod
    def from_string(target_string: str) -> Target:
        items = target_string.split()
        name, attrs = items[0], items[1:]
        return Target(name, attrs)


class Prologue(Node):
    def __init__(self, extra_inputs, indices, value):
        self.extra_inputs: List[TensorNode] = extra_inputs
        self.indices: List[Var] = indices
        self.value: Expr = value


class Epilogue(Node):
    def __init__(self, extra_inputs, indices, orig_value, value, out_indices, out_tensor):
        self.extra_inputs: List[TensorNode] = extra_inputs
        self.indices: List[Var] = indices
        self.orig_value: Var = orig_value
        self.value: Expr = value
        self.out_indices: List[Expr] = out_indices
        self.out_tensor: TensorNode = out_tensor


class TaskContext:
    contexts = []

    def __init__(self, space_level: int = 0, resolve_out_dir: str = None):
        self.space_level = space_level
        self.resolve_out_dir = resolve_out_dir

    def __enter__(self):
        self.contexts.append(self)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.contexts.pop()

    @staticmethod
    def current() -> TaskContext:
        return TaskContext.contexts[-1]


class InverseMap:
    def __init__(self, axes: List[Var], indices: List[Expr]):
        self.axes: List[Var] = axes
        self.indices: List[Expr] = indices

    @staticmethod
    def from_lambda(func, num_args=None) -> InverseMap:
        num_args = num_args if num_args is not None else func.__code__.co_argcount
        axes = [var('v') for v in range(num_args)]
        indices = list(func(*axes))
        return InverseMap(axes, indices)

    def __add__(self, other) -> InverseMap:
        from hidet.ir.functors import rewrite
        if not isinstance(other, InverseMap):
            raise ValueError('Can not concat InverseMap with {}'.format(type(other)))
        lhs, rhs = self, other
        if len(lhs.indices) != len(rhs.axes):
            raise ValueError('Can not concat InverseMap a and b, '
                             'where a has {} indices and b has {} axes'.format(len(lhs.indices), len(rhs.axes)))
        rmap = {a: b for a, b in zip(rhs.axes, lhs.indices)}
        indices = [rewrite(index_expr, rmap) for index_expr in rhs.indices]
        return InverseMap(lhs.axes, indices)


class Task(Node):
    def __init__(self, name, inputs, outputs, prologues=None, epilogues=None, parameters=None, inverse_map=None):
        self.name = name
        self.inputs: List[TensorNode] = inputs
        self.outputs: List[TensorNode] = outputs
        self.prologues: Dict[TensorNode, Prologue] = prologues if prologues else {}
        self.epilogues: Dict[TensorNode, Epilogue] = epilogues if epilogues else {}
        self.parameters: List[TensorNode] = parameters if parameters else inputs + outputs

        inverse_map = inverse_map if inverse_map else {}
        self.inverse_map: Dict[TensorNode, InverseMap] = {
            a: (b if isinstance(b, InverseMap) else InverseMap.from_lambda(b)) for a, b in inverse_map.items()
        }

        sanity_check(self)

    def implement(self, target: Union[Target, str]) -> IRModule:
        if isinstance(target, str):
            target = Target.from_string(target)
        if target.name == 'cuda':
            ret = self.implement_cuda()
        elif target.name == 'cpu':
            ret = self.implement_cpu()
        else:
            raise ValueError()
        if not isinstance(ret, IRModule):
            raise AssertionError('The task implement function should return an IRModule, but got a {}.'.format(type(ret)))
        return ret

    def implement_cuda(self) -> IRModule:
        from hidet.tos.ops.schedules import generic_cuda_schedule
        return generic_cuda_schedule(self)

    def implement_cpu(self) -> IRModule:
        from hidet.tos.ops.schedules import generic_cpu_schedule
        return generic_cpu_schedule(self)


def is_elementwise(task: Task) -> bool:
    """
    A task is elementwise if and only if there is only no reduce compute in the task.

    Parameters
    ----------
    task: Task
        The task to check.

    Returns
    -------
    ret: bool
        Whether the task is elementwise.
    """
    from hidet.ir.functors import collect
    scalar_nodes: List[ScalarNode] = collect(task.outputs, ScalarNode, stop_when_found=False)
    return all(sn.reduce_compute is None for sn in scalar_nodes)


def is_unary_elementwise(task: Task) -> bool:
    return len(task.inputs) == 1 and len(task.outputs) == 1 and is_elementwise(task)


def sanity_check(task: Task):
    from hidet.ir.functors import collect
    # todo: check
    pass
