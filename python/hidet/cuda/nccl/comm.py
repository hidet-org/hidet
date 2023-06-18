from enum import IntEnum
from typing import List
import struct

from .ffi import nccl_runtime_api, NcclUniqueId
from hidet.ffi.utils import Array
from hidet.ir.type import void_p

class ncclDataType(IntEnum):	
    int8 = 0	
    char = 0	
    uint8 = 1	
    int32 = 2	
    int = 2	
    uint32 = 3	
    int64 = 4	
    uint64 = 5	
    float16 = 6	
    half = 6	
    float32 = 7	
    float = 7	
    float64 = 8	
    double = 8	
    bfloat = 9	

class ncclRedOp(IntEnum):	
    sum = 0	
    prod = 1	
    max = 2	
    min = 3	
    avg = 4	

class NcclCommunicator:
    def __init__(self, handle: int):
        """
        Users should not call this constructor directly. Because there are two ways of creating
        a new communicator: 1) using unique_id and rank ; 2) using split.
        """

        self._handle = handle
    
    def __del__(self):
        nccl_runtime_api.comm_destroy(self._handle)

    def split(self):
        raise NotImplementedError()
    
def create_comm(nranks: int, unique_id: NcclUniqueId, rank: int):
    handle = nccl_runtime_api.comm_init_rank(nranks, unique_id, rank)
    return NcclCommunicator(handle)

def comms_to_array(comms: List[NcclCommunicator]):
    handles = [comm._handle for comm in comms]
    array = Array(void_p, len(comms))
    struct.pack_into(array.format, array.buffer, 0, *handles)
    return array