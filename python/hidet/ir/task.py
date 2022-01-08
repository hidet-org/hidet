from typing import Union, Optional, List, Dict
from hidet.ir.expr import Var
from hidet.ir.node import Node
from hidet.ir.dialects.compute import ScalarInput, TensorInput, ComputeNode
from hidet.ir.type import BaseType
from hidet.ir.expr import Expr


class Worker(Node):
    pass


class Grid(Worker):
    def __init__(self, grid_dim=None, block_dim=None):
        self.grid_dim: Optional[Expr] = grid_dim
        self.block_dim: Optional[Expr] = block_dim


class ThreadBlock(Worker):
    def __init__(self, block_dim=None):
        self.block_dim: Optional[Expr] = block_dim


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

