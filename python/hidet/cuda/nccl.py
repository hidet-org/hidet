from ..ffi import nccl_runtime_api, NcclUniqueId
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