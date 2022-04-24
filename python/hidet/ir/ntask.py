from typing import Dict, List, Optional
from hidet.ir.node import Node
from hidet.ir.expr import Expr
from hidet.ir.func import IRModule
from hidet.ir.dialects.compute import TensorCompute, TensorInput, ComputeNode
from hidet.ir.type import TensorType


class Prologue(Node):
    def __init__(self):
        self.extra_inputs: List[TensorInput] = []

    def fetch(self, *indices) -> Expr:
        raise NotImplementedError()


class Epilogue(Node):
    def __init__(self):
        self.extra_inputs: List[TensorInput] = []

    def forward(self, value, *indices) -> Expr:
        raise NotImplementedError()


class Task(Node):
    def __init__(self, name, inputs, outputs, prologues=None, epilogues=None, parameters=None, schedulers=None):
        self.name = name
        self.inputs: List[TensorInput] = inputs
        self.outputs: List[TensorCompute] = outputs
        self.prologues: Dict[TensorInput, Prologue] = prologues if prologues else {}
        self.epilogues: Dict[TensorInput, Epilogue] = epilogues if epilogues else {}
        self.parameters: List[ComputeNode] = parameters if parameters else inputs + outputs
        self.schedulers: Dict[str, Scheduler] = schedulers if schedulers else {}


class Scheduler(Node):
    def __call__(self, task: Task) -> IRModule:
        return self.schedule(task)

    def schedule(self, task: Task) -> IRModule:
        raise NotImplementedError()
