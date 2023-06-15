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
from typing import Union
from ctypes import c_void_p, c_char_p, c_uint64, c_int32, c_int, pointer, Structure, c_byte
from enum import IntEnum
from hidet.cuda import Stream
from .ffi import get_func, nccl_available, _LIB_NCCL


class RuntimeAPI:
    _set_current_stream = get_func('set_cuda_stream', [c_void_p], None)
    _get_current_stream = get_func('get_cuda_stream', [], c_void_p)
    _register_callback = get_func('register_callback', [c_char_p, c_void_p], None)
    _allocate_cuda_storage = get_func('allocate_cuda_storage', [c_uint64], c_uint64)
    _free_cuda_storage = get_func('free_cuda_storage', [c_uint64], None)
    _reset_symbol_table = get_func('reset_symbol_table', [], None)
    _get_symbol_value = get_func('get_symbol_value', [c_char_p], c_int32)
    _set_symbol_value = get_func('set_symbol_value', [c_char_p, c_int32], None)

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

runtime_api = RuntimeAPI()

if nccl_available():
    print("NCCL is available")

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

    class NCCLRuntimeAPI:
        """
        Runtime APIs regarding NCCL
        TODO: Exception handling
        """
        _get_version = get_func('ncclGetVersion', [c_void_p], c_int)
        _get_unique_id = get_func('ncclGetUniqueId', [c_void_p], c_int)
        _comm_init_rank = get_func('ncclCommInitRank', [c_void_p, c_int, NcclUniqueId, c_int], c_int)
        _comm_destroy = get_func('ncclCommDestroy', [c_void_p], c_int)
        _all_reduce = get_func('ncclAllReduce', [c_void_p, c_void_p, c_uint64, c_int, c_int, c_void_p, c_int], c_int)
        _broadcast = get_func('ncclBroadcast', [c_void_p, c_void_p, c_uint64, c_int, c_int, c_void_p, c_int], c_int)
        _reduce = get_func('ncclReduce', [c_void_p, c_void_p, c_uint64, c_int, c_int, c_int, c_void_p, c_int], c_int)
        _all_gather = get_func('ncclAllGather', [c_void_p, c_void_p, c_uint64, c_int, c_void_p, c_int], c_int)
        _reduce_scatter = get_func('ncclReduceScatter', [c_void_p, c_void_p, c_uint64, c_int, c_int, c_void_p, c_int], c_int)

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
        def comm_destroy(comm_handle:int) -> None:
            ret = NCCLRuntimeAPI._comm_destroy(comm_handle)
            assert ret == 0
            
        @staticmethod
        def all_reduce(sendbuff:int, recvbuff:int, count:int, datatype:ncclDataType, op:ncclRedOp, comm_handle:int, s:Stream) -> None:
            print(type(datatype), op)
            ret = NCCLRuntimeAPI._all_reduce(
                sendbuff, recvbuff, count, datatype, op, comm_handle, s._handle
            )
            assert ret == 0


        


    nccl_runtime_api = NCCLRuntimeAPI()