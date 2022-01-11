from typing import Union, Optional, List, Dict
from hidet.ir.node import Node
from hidet.ir.dialects.compute import ScalarInput, TensorInput, ComputeNode
from hidet.ir.type import BaseType
from hidet.ir.expr import Expr, convert


class Worker(Node):
    pass


class Grid(Worker):
    def __init__(self, grid_dim=None, block_dim=None):
        self.grid_dim: Optional[Expr] = convert(grid_dim) if grid_dim else None
        self.block_dim: Optional[Expr] = convert(block_dim) if block_dim else None


class ThreadBlock(Worker):
    def __init__(self, block_dim=None):
        self.block_dim: Optional[Expr] = convert(block_dim) if block_dim else None


class Warp(Worker):
    def __init__(self):
        pass


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
        self.params_type: Dict[Union[ScalarInput, TensorInput, ComputeNode], BaseType] = params_type
        self.worker: Worker = worker

