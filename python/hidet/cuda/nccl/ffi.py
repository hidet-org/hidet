# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional
import ctypes
from ctypes import c_void_p, c_int, pointer, Structure, c_byte, POINTER, c_uint64
import glob
import os

from hidet.ffi.ffi import get_func
from hidet.cuda import Stream
from .libinfo import get_nccl_library_search_dirs

_LIB_NCCL: Optional[ctypes.CDLL] = None
nccl_library_path = None


class NcclUniqueId(Structure):
    """
    Defined as in nccl.h
    """

    _fields_ = [("internal", c_byte * 128)]


def nccl_available():
    return _LIB_NCCL is not None


def nccl_version():
    return nccl_runtime_api.get_version()


def load_nccl_library():
    global _LIB_NCCL, nccl_library_path
    library_dirs = get_nccl_library_search_dirs()
    for library_dir in library_dirs:
        lib_nccl_paths = glob.glob(os.path.join(library_dir, 'libnccl.so*'))
        if len(lib_nccl_paths) == 0:
            continue
        _LIB_NCCL = ctypes.cdll.LoadLibrary(lib_nccl_paths[0])
        nccl_library_path = lib_nccl_paths[0]
        break


load_nccl_library()


def nccl_library_filename():
    return os.path.basename(nccl_library_path)


if nccl_available():

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

        _all_reduce = get_func(
            'ncclAllReduce', [c_void_p, c_void_p, c_uint64, c_int, c_int, c_void_p, c_int], c_int, lib=_LIB_NCCL
        )
        _broadcast = get_func(
            'ncclBroadcast', [c_void_p, c_void_p, c_uint64, c_int, c_int, c_void_p, c_int], c_int, lib=_LIB_NCCL
        )
        _reduce = get_func(
            'ncclReduce', [c_void_p, c_void_p, c_uint64, c_int, c_int, c_int, c_void_p, c_int], c_int, lib=_LIB_NCCL
        )
        _all_gather = get_func(
            'ncclAllGather', [c_void_p, c_void_p, c_uint64, c_int, c_void_p, c_int], c_int, lib=_LIB_NCCL
        )
        _reduce_scatter = get_func(
            'ncclReduceScatter', [c_void_p, c_void_p, c_uint64, c_int, c_int, c_void_p, c_int], c_int, lib=_LIB_NCCL
        )

        # Early versions of NCCL do not have split
        try:
            _comm_split = get_func('ncclCommSplit', [c_void_p, c_int, c_int, c_void_p, c_void_p], c_int, lib=_LIB_NCCL)
        except ValueError:
            _comm_split = None

        @staticmethod
        def get_version() -> int:
            version = c_int(0)
            NCCLRuntimeAPI._get_version(pointer(version))
            return version.value

        @staticmethod
        def get_unique_id(comm_id: NcclUniqueId) -> None:
            """
            In-place initialization of the NcclUniqueId object
            """
            ret = NCCLRuntimeAPI._get_unique_id(pointer(comm_id))
            assert ret == 0, ret

        @staticmethod
        def comm_init_rank(ndev: int, comm_id: NcclUniqueId, rank: int) -> int:
            comm = c_void_p()
            ret = NCCLRuntimeAPI._comm_init_rank(pointer(comm), ndev, comm_id, rank)
            assert ret == 0, ret
            return comm.value

        @staticmethod
        def comm_destroy(comm_handle) -> None:
            ret = NCCLRuntimeAPI._comm_destroy(comm_handle)
            assert ret == 0

        @staticmethod
        def comm_split(comm_handle: int, color: int, key: int) -> int:
            if NCCLRuntimeAPI._comm_split is None:
                raise RuntimeError("split is not supported on this version of NCCL. Please install a newer version.")
            comm = c_void_p()
            ret = NCCLRuntimeAPI._comm_split(comm_handle, color, key, pointer(comm), None)
            assert ret == 0
            return comm.value

        # TODO: Currently only support all_reduce
        @staticmethod
        def all_reduce(
            sendbuff: int, recvbuff: int, count: int, datatype: int, op: int, comm_handle: int, s: Stream
        ) -> None:
            ret = NCCLRuntimeAPI._all_reduce(sendbuff, recvbuff, count, datatype, op, comm_handle, s.handle())
            assert ret == 0

    nccl_runtime_api = NCCLRuntimeAPI()
