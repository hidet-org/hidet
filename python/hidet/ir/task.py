from typing import Union, Optional, List, Dict
from hidet.ir.node import Node
from hidet.ir.dialects.compute import ScalarInput, TensorInput, ComputeNode
from hidet.ir.type import TypeNode
from hidet.ir.expr import Expr, convert
from hidet.ir.layout import TaskLayout

Int = Union[Expr, int]


class Worker(Node):
    pass


class Grid(Worker):
    def __init__(self, grid_dim: Optional[Int] = None, block_dim: Optional[Int] = None, min_blocks: Optional[Int] = None):
        self.grid_dim: Optional[Expr] = convert(grid_dim) if grid_dim else None
        self.block_dim: Optional[Expr] = convert(block_dim) if block_dim else None
        self.min_blocks: Optional[Expr] = convert(min_blocks) if min_blocks else None


class ThreadBlock(Worker):
    def __init__(self, block_dim: Optional[Int] = None, task_layout: Optional[TaskLayout] = None):
        self.block_dim: Optional[Expr] = convert(block_dim) if block_dim else None
        self.task_layout: Optional[TaskLayout] = task_layout


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
    def __init__(self, name, computation, params, params_type, worker):
        self.name: str = name
        self.compute: ComputeNode = computation
        self.params: List[Union[ScalarInput, TensorInput, ComputeNode]] = params
        self.params_type: Dict[Union[ScalarInput, TensorInput, ComputeNode], TypeNode] = params_type
        self.worker: Worker = worker
