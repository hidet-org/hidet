from typing import Optional, List
import ctypes
from ctypes import c_void_p, c_char_p, c_uint64, c_int32, c_int, pointer, Structure, c_byte, POINTER
from enum import IntEnum
import glob, os
from functools import partial

from hidet.ffi.ffi import get_func
from .libinfo import get_nccl_library_search_dirs

_LIB_NCCL: Optional[ctypes.CDLL] = None

class NcclUniqueId(Structure):
    """
    Defined in nccl.h
    """
    _fields_ = [("internal", c_byte * 128)]

def nccl_available():
    return _LIB_NCCL is not None

def load_nccl_library():
    global _LIB_NCCL
    library_dirs = get_nccl_library_search_dirs()
    for library_dir in library_dirs:
        lib_nccl_paths = glob.glob(os.path.join(library_dir, 'libnccl.so*'))
        if len(lib_nccl_paths) == 0:
            continue
        _LIB_NCCL = ctypes.cdll.LoadLibrary(lib_nccl_paths[0])
        nccl_library_paths = lib_nccl_paths[0]
        break
    if _LIB_NCCL is None:
        raise OSError('Can not find nccl library in the following directory: \n' + '\n'.join(library_dirs))

load_nccl_library()
if not nccl_available():
    raise RuntimeError("NCCL Library not found.")

class NCCLRuntimeAPI:
    """
    Runtime APIs regarding NCCL
    TODO: Exception handling
    """
    _get_version = get_func('ncclGetVersion', [c_void_p], c_int, lib=_LIB_NCCL)
    _get_unique_id = get_func('ncclGetUniqueId', [c_void_p], c_int, lib=_LIB_NCCL)
    _comm_init_rank = get_func('ncclCommInitRank', [c_void_p, c_int, NcclUniqueId, c_int], c_int, lib=_LIB_NCCL)
    _comm_destroy = get_func('ncclCommDestroy', [c_void_p], c_int, lib=_LIB_NCCL)

    _comm_user_rank = get_func('ncclCommUserRank', [c_void_p, POINTER(c_int)], c_int, lib=_LIB_NCCL)
    _comm_count = get_func('ncclCommCount', [c_void_p, POINTER(c_int)], c_int, lib=_LIB_NCCL)

    @staticmethod
    def get_version() -> int:
        version = c_int(0)
        NCCLRuntimeAPI._get_version(pointer(version))
        return version.value
    
    @staticmethod
    def get_unique_id(comm_id:NcclUniqueId) -> None:
        """
        In-place initialization of the NcclUniqueId object
        """
        ret = NCCLRuntimeAPI._get_unique_id(pointer(comm_id))
        assert ret == 0, ret
    
    @staticmethod
    def comm_init_rank(ndev:int, comm_id:NcclUniqueId, rank:int) -> int:
        comm = c_void_p()
        ret = NCCLRuntimeAPI._comm_init_rank(pointer(comm), ndev, comm_id, rank)
        assert ret == 0, ret
        return comm.value
    
    @staticmethod
    def comm_destroy(comm_handle) -> None:
        ret = NCCLRuntimeAPI._comm_destroy(comm_handle)
        assert ret == 0

nccl_runtime_api = NCCLRuntimeAPI()

