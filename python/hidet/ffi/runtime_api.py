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
from ctypes import c_void_p, c_char_p, c_uint64, c_int32
from hidet.cuda import Stream
from .ffi import get_func
from .utils import Array


class RuntimeAPI:
    _set_current_stream = get_func('set_cuda_stream', [c_void_p], None)
    _get_current_stream = get_func('get_cuda_stream', [], c_void_p)
    _register_callback = get_func('register_callback', [c_char_p, c_void_p], None)
    _allocate_cuda_storage = get_func('allocate_cuda_storage', [c_uint64], c_uint64)
    _free_cuda_storage = get_func('free_cuda_storage', [c_uint64], None)
    _reset_symbol_table = get_func('reset_symbol_table', [], None)
    _get_symbol_value = get_func('get_symbol_value', [c_char_p], c_int32)
    _set_symbol_value = get_func('set_symbol_value', [c_char_p, c_int32], None)
    _set_nccl_comms = get_func('set_nccl_comms', [c_int32, c_void_p], None)

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
    def set_nccl_comms(comms: Array) -> None:
        comms_array_t = c_void_p * comms.length
        RuntimeAPI._set_nccl_comms(comms.length, comms_array_t.from_buffer(comms.buffer))


runtime_api = RuntimeAPI()
