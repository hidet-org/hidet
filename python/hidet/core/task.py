from typing import Union, Optional, List, Dict
from hidet.ir.expr import Var
from hidet.ir.type import Type
from hidet.core.compute import ScalarInput, TensorInput, ComputeNode
from hidet.core.worker import Worker


class Task:
    def __init__(self, name, compute, params, params_type, worker):
        self.name: str = name
        self.compute: ComputeNode = compute
        self.params: List[Union[ScalarInput, TensorInput, ComputeNode]] = params
        self.params_type: Dict[Union[ScalarInput, TensorInput, ComputeNode], Type] = params_type
        self.worker: Worker = worker

