from __future__ import annotations
from typing import Dict, List, Union, Optional, Sequence, Type
from hidet.ir.node import Node
from hidet.ir.expr import Expr, Var
from hidet.ir.func import IRModule
from hidet.ir.dialects.compute import TensorCompute, TensorInput, ComputeNode
from hidet.utils.doc import Doc, Text, doc_join, NewLine


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
        self.extra_inputs: List[TensorInput] = extra_inputs
        self.indices: List[Var] = indices
        self.value: Expr = value


class Epilogue(Node):
    def __init__(self, extra_inputs, indices, value):
        self.extra_inputs: List[TensorInput] = extra_inputs
        self.indices: List[Var] = indices
        self.value: Expr = value


class ImplementContext:
    contexts: List['ImplementContext'] = []

    def __init__(self, space_level=0):
        self.space_level = space_level

    def __enter__(self):
        ImplementContext.contexts.append(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        assert ImplementContext.contexts[-1] is self
        ImplementContext.contexts.pop()

    @classmethod
    def current(cls):
        return cls.contexts[-1]


ImplementContext.contexts.append(ImplementContext())    # fallback context


class Task(Node):
    def __init__(self, name, inputs, outputs, prologues=None, epilogues=None, parameters=None):
        self.name = name
        self.inputs: List[TensorInput] = inputs
        self.outputs: List[TensorCompute] = outputs
        self.prologues: Dict[TensorInput, Prologue] = prologues if prologues else {}
        self.epilogues: Dict[TensorInput, Epilogue] = epilogues if epilogues else {}
        self.parameters: List[ComputeNode] = parameters if parameters else inputs + outputs

    def __str__(self):
        lines = [
            Text('name: ') + self.name,
            Text('inputs: ') + '[' + doc_join(['{}: {}'.format(v, v.data_type) for v in self.inputs], ', ') + ']',
            Text('outputs: ') + '[' + doc_join([str(v) for v in self.outputs], ', ') + ']',
            Text('parameters: ') + '[' + doc_join([v.name for v in self.parameters], ', ') + ']'
        ]
        doc = Text('Task(') + doc_join(lines, NewLine()).indent() + ')'
        return str(doc)

    # todo: remove this after refactering
    @property
    def compute(self):
        return self.outputs[0]

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

