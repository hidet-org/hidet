from ctypes import c_void_p, c_char_p, c_uint64, c_int32, c_int, pointer, Structure, c_byte, POINTER
from enum import IntEnum

from hidet.ffi import runtime_api

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

class NcclUniqueId(Structure):
    """
    Defined in nccl.h
    """
    _fields_ = [("internal", c_byte * 128)]

class NcclCommunicator:
    def __init__(self, handle: int):
        """
        Users should not call this constructor directly. Because there are two ways of creating
        a new communicator: 1) using unique_id and rank ; 2) using split.
        """

        self._handle = handle
        runtime_api.add_nccl_comm(handle)
    
    def __del__(self):
        from hidet.cuda.nccl.ffi import nccl_runtime_api
        nccl_runtime_api.comm_destroy(self)

    def split(self):
        raise NotImplementedError()
    
def create_comm(nranks: int, unique_id: NcclUniqueId, rank: int):
    from hidet.cuda.nccl.ffi import nccl_runtime_api
    handle = nccl_runtime_api.comm_init_rank(nranks, unique_id, rank)
    return NcclCommunicator(handle)