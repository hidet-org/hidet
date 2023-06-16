from enum import IntEnum

from ..ffi import NcclUniqueId, NcclCommunicator, nccl_runtime_api

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

def create_comm(nranks: int, unique_id: NcclUniqueId, rank: int):
    handle = nccl_runtime_api.comm_init_rank(nranks, unique_id, rank)
    return NcclCommunicator(handle)