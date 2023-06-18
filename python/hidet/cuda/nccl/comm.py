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
from enum import IntEnum
from typing import List
import struct

from hidet.ffi.utils import Array
from hidet.ir.type import void_p, DataType
from .ffi import nccl_runtime_api, NcclUniqueId


class NcclDataType(IntEnum):
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


class NcclRedOp(IntEnum):
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

    @property
    def handle(self):
        return self._handle

    def split(self):
        raise NotImplementedError()


def create_comm(nranks: int, unique_id: NcclUniqueId, rank: int) -> NcclCommunicator:
    handle = nccl_runtime_api.comm_init_rank(nranks, unique_id, rank)
    return NcclCommunicator(handle)


def comms_to_array(comms: List[NcclCommunicator]) -> Array:
    handles = [comm.handle for comm in comms]
    array = Array(void_p, len(comms))
    struct.pack_into(array.format, array.buffer, 0, *handles)
    return array


def init_unique_id(unqie_id: NcclUniqueId) -> None:
    nccl_runtime_api.get_unique_id(unqie_id)


def dtype_to_nccl(dtype: DataType) -> NcclDataType:
    sname_dict = {
        'f64': NcclDataType.float64,
        'f32': NcclDataType.float32,
        'bf16': NcclDataType.bfloat,
        'f16': NcclDataType.float16,
        'i64': NcclDataType.int64,
        'i32': NcclDataType.int32,
        'i8': NcclDataType.int8,
        'u64': NcclDataType.uint64,
        'u32': NcclDataType.uint32,
        'u8': NcclDataType.uint8,
    }
    sname = dtype.short_name
    nccl_type = sname_dict.get(sname, None)
    if nccl_type is None:
        raise RuntimeError(f"Data type {dtype.name} is not supported in NCCL")
    return nccl_type
