from typing import Union, Optional, List, Dict
from hidet.ir.node import Node
from hidet.ir.dialects.compute import ScalarInput, TensorInput, ComputeNode
from hidet.ir.type import TypeNode, TensorType, ScalarType
from hidet.ir.expr import Expr, convert
from hidet.ir.layout import TaskLayout

Int = Union[Expr, int]


class Worker(Node):
    pass


class Grid(Worker):
    def __init__(self, grid_dim: Optional[Int] = None, block_dim: Optional[Int] = None, dynamic_smem_bytes: Optional[Int] = 0, min_blocks: Optional[Int] = None):
        self.grid_dim: Optional[Expr] = convert(grid_dim)
        self.block_dim: Optional[Expr] = convert(block_dim)
        self.min_blocks: Optional[Expr] = convert(min_blocks)
        self.dynamic_smem_bytes: Optional[Expr] = convert(dynamic_smem_bytes)


class ThreadBlock(Worker):
    def __init__(self, block_dim: Optional[Int] = None, task_layout=None):
        self.block_dim: Optional[Expr] = convert(block_dim) if block_dim else None
        self.task_layout: Optional[TaskLayout] = task_layout

        if self.block_dim is None and self.task_layout is not None:
            # infer block size from task layout
            self.block_dim = convert(self.task_layout.num_workers)


class Warp(Worker):
    def __init__(self, task_layout: Optional[TaskLayout] = None):
        self.task_layout: Optional[TaskLayout] = task_layout


class Thread(Worker):
    def __init__(self):
        pass


class Host(Worker):
    def __init__(self):
        pass


class Task(Node):
    def __init__(self, name, computation, params, worker):
        self.name: str = name
        self.compute: ComputeNode = computation
        self.params: List[Union[ScalarInput, TensorInput, ComputeNode]] = params
        self.worker: Worker = worker

    def __str__(self):
        from hidet.utils.doc import Doc, Text, NewLine
        body = Doc()
        body += NewLine() + 'name: ' + self.name
        body += NewLine() + 'compute: ' + str(self.compute)
        body += NewLine() + 'params: ' + ', '.join(['{}: {}'.format(param.name, param.data_type()) for param in self.params])
        body += NewLine() + 'worker: ' + str(self.worker)
        return str('Task(' + body.indent() + NewLine() + ')')

    def param_types(self) -> List[Union[ScalarType, TensorType]]:
        return [param.data_type() for param in self.params]

    def type_of_param(self, given_param) -> Union[TypeNode, TensorType]:
        for param, param_type in zip(self.params, self.param_types()):
            if given_param is param:
                return param_type
        raise KeyError()
