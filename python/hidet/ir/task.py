from __future__ import annotations
from typing import Dict, List, Union, Optional, Sequence, Type, Tuple
from hidet.ir.node import Node
from hidet.ir.expr import Expr, Var
from hidet.ir.func import IRModule
from hidet.ir.dialects.compute import TensorNode


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
        self.out_indices: Optional[List[Expr]] = out_indices
        self.out_tensor: Optional[TensorNode] = out_tensor


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


class Task(Node):
    def __init__(self, name, inputs, outputs, prologues=None, epilogues=None, parameters=None, reverse_map=None):
        self.name = name
        self.inputs: List[TensorNode] = inputs
        self.outputs: List[TensorNode] = outputs
        self.prologues: Dict[TensorNode, Prologue] = prologues if prologues else {}
        self.epilogues: Dict[TensorNode, Epilogue] = epilogues if epilogues else {}
        self.parameters: List[TensorNode] = parameters if parameters else inputs + outputs
        self.reverse_map: Optional[Tuple[List[Var], List[Expr]]] = reverse_map

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

