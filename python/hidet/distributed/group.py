import hidet
from hidet.graph import Tensor
from typing import Optional, List

from .store import Store, FileStore
from hidet.cuda.nccl import create_unique_id, NcclUniqueId, create_comm, NcclCommunicator, comms_to_array

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

    def barrier(self):
        raise NotImplementedError()

NCCL_COMMS = []

class NCCLProcessGroup(ProcessGroup):
    def __init__(self, comm: NcclCommunicator, world_size: int, rank: int):
        global NCCL_COMMS
        self._comm: NcclCommunicator = comm
        self._world_size: int = world_size
        self._rank: int = rank
        NCCL_COMMS.append(comm)
    
    def rank(self) -> int:
        return self._rank
    
    def size(self) -> int:
        return self._world_size
    
    def all_reduce(self, tensor: Tensor, op:str):
        assert not tensor.is_symbolic()
        assert tensor.device.is_cuda()
        addr = tensor.storage.addr
        self._comm.all_reduce(addr, addr, tensor.nbytes, tensor.dtype, op)

def create_nccl_group(store: Store, world_size: int, rank: int):
    if rank == 0:
        unique_id = create_unique_id()
        store.set('unique_id', unique_id.internal)
    else:
        unique_id = store.get('unique_id')
        unique_id = NcclUniqueId(unique_id)
    comm = create_comm(world_size, unique_id, rank)
    return NCCLProcessGroup(comm, world_size, rank)

def set_nccl_comms():
    from hidet.ffi.runtime_api import runtime_api
    comm_array = comms_to_array(NCCL_COMMS)
    runtime_api.set_nccl_comms(comm_array)


if __name__ == '__main__':
    store = FileStore('tmp')
    group = create_nccl_group(store, 1, 0)