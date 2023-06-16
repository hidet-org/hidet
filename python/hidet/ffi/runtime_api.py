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
from typing import Union, Optional, List
from ctypes import c_void_p, c_char_p, c_uint64, c_int32, c_int, pointer, Structure, c_byte, POINTER
from enum import IntEnum
from hidet.cuda import Stream
from .ffi import get_func, nccl_available

class NcclUniqueId(Structure):
    """
    Defined in nccl.h
    """
    _fields_ = [("internal", c_byte * 128)]

class NcclCommunicator:
    """

    """
    def __init__(self, handle: int):
        """
        Users should not call this constructor directly. Because there are two ways of creating
        a new communicator: 1) using unique_id and rank ; 2) using split.
        """
        if not nccl_available():
            raise RuntimeError("NCCL Library not found.")
        self._handle = handle
        _comms.append(self)
    
    def __del__(self):
        """
        Should we manage the lifetime of communicator object in Python or C++?
        """
        nccl_runtime_api.comm_destroy(self)

    def split(self):
        raise NotImplementedError()

class RuntimeAPI:
    _set_current_stream = get_func('set_cuda_stream', [c_void_p], None)
    _get_current_stream = get_func('get_cuda_stream', [], c_void_p)
    _register_callback = get_func('register_callback', [c_char_p, c_void_p], None)
    _allocate_cuda_storage = get_func('allocate_cuda_storage', [c_uint64], c_uint64)
    _free_cuda_storage = get_func('free_cuda_storage', [c_uint64], None)
    _reset_symbol_table = get_func('reset_symbol_table', [], None)
    _get_symbol_value = get_func('get_symbol_value', [c_char_p], c_int32)
    _set_symbol_value = get_func('set_symbol_value', [c_char_p, c_int32], None)
    _add_nccl_comm = get_func('add_nccl_comm', [c_void_p], None)
    _get_nccl_comm = get_func('get_nccl_comm', [], c_void_p)

    @staticmethod
    def set_current_stream(stream: Union[Stream, int]) -> None:
        RuntimeAPI._set_current_stream(c_void_p(int(stream)))

    @staticmethod
    def get_current_stream() -> int:
        p = RuntimeAPI._get_current_stream()
        return p.value

    @staticmethod
    def register_callback(name: str, cfunc):
        name = name.encode('utf-8')
        RuntimeAPI._register_callback(name, cfunc)

    @staticmethod
    def allocate_cuda_storage(nbytes: int) -> int:
        return RuntimeAPI._allocate_cuda_storage(nbytes)

    @staticmethod
    def free_cuda_storage(addr: int) -> None:
        return RuntimeAPI._free_cuda_storage(addr)

    @staticmethod
    def reset_symbol_table() -> None:
        RuntimeAPI._reset_symbol_table()

    @staticmethod
    def get_symbol_value(name: str) -> int:
        name = name.encode('utf-8')
        return RuntimeAPI._get_symbol_value(name)

    @staticmethod
    def set_symbol_value(name: str, value: int) -> None:
        name = name.encode('utf-8')
        RuntimeAPI._set_symbol_value(name, value)

    @staticmethod
    def add_nccl_comm(comm: NcclCommunicator) -> None:
        if not nccl_available():
            raise RuntimeError("NCCL Library not found.")
        RuntimeAPI._add_nccl_comm(comm._handle)
    
    @staticmethod
    def get_nccl_comm(comm_id: int) -> NcclCommunicator:
        if not nccl_available():
            raise RuntimeError("NCCL Library not found.") 
        comm_handle = RuntimeAPI._get_nccl_comm(comm_id)
        return NcclCommunicator(comm_handle)

runtime_api = RuntimeAPI()

if nccl_available():
    class NCCLRuntimeAPI:
        """
        Runtime APIs regarding NCCL
        TODO: Exception handling
        """
        _get_version = get_func('ncclGetVersion', [c_void_p], c_int)
        _get_unique_id = get_func('ncclGetUniqueId', [c_void_p], c_int)
        _comm_init_rank = get_func('ncclCommInitRank', [c_void_p, c_int, NcclUniqueId, c_int], c_int)
        _comm_destroy = get_func('ncclCommDestroy', [c_void_p], c_int)

        _comm_user_rank = get_func('ncclCommUserRank', [c_void_p, POINTER(c_int)], c_int)
        _comm_count = get_func('ncclCommCount', [c_void_p, POINTER(c_int)], c_int)

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
        def comm_destroy(comm:NcclCommunicator) -> None:
            ret = NCCLRuntimeAPI._comm_destroy(comm._handle)
            assert ret == 0

    nccl_runtime_api = NCCLRuntimeAPI()
    _comms: List[NcclCommunicator] = []

    def get_nccl_comm(comm_id: int):
        return _comms[comm_id]
