import hidet
from hidet import Tensor
from typing import Optional, List

from .store import Store

class ProcessGroup:
    def backend(self) -> str:
        raise NotImplementedError()
    
    def rank(self) -> int:
        raise NotImplementedError()
    
    def size(self) -> int:
        raise NotImplementedError()
    
    def broadcast(self, tensor: Tensor, src: int):
        raise NotImplementedError()
    
    def all_reduce(self, tensor: Tensor, op: str):
        raise NotImplementedError()
    
    def reduce(self, tensor: Tensor, dst:int, op:str):
        raise NotImplementedError()
    
    def all_gather(self, tensor_list: List[Tensor], tensor: Tensor):
        raise NotImplementedError()
    
    def all_gather_into_tensor(self, output_tensor: Tensor, input_tensor: Tensor):
        raise NotImplementedError()
    
    def gather(self, tensor: Tensor, gather_list: Optional[List[Tensor]]=None, dst: int=0):
        raise NotImplementedError()
    
    def scatter(self, tensor: Tensor, scattler_list: Optional[List[Tensor]]=None):
        raise NotImplementedError()
    
    def reduce_scatter(self, output: Tensor, input_list: List[Tensor], op: str):
        raise NotImplementedError()
    
    def reduce_scatter_tensor(self, output: Tensor, input: Tensor, op: str):
        raise NotImplementedError()

class NCCLProcessGroup(ProcessGroup):
    def __init__(self, store: Store, world_size: int, rank: int):
        if rank == 0