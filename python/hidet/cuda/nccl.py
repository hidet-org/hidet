from ..ffi import nccl_runtime_api, NcclUniqueId, ncclDataType, ncclRedOp
from hidet.cuda import Stream
from ctypes import c_int, c_void_p, pointer

class NcclCommunicator:
    """
    I copy the api from cupy
    """
    def __init__(self, ndev:int, commId:NcclUniqueId, rank:int):
        self._ndev = ndev
        self._commId = commId
        self._rank = rank

        self._handle = nccl_runtime_api.comm_init_rank(ndev, commId, rank)
    
    def __del__(self):
        nccl_runtime_api.comm_destroy(self._handle)
    
    def all_reduce(self, sendbuff:int, recvbuff:int, count:int, datatype:ncclDataType, op:ncclRedOp, s:Stream):
        nccl_runtime_api.all_reduce(sendbuff, recvbuff, count, datatype, op, self._handle, s)