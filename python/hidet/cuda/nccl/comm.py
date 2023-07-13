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
from typing import List, Optional
import struct

from hidet.ffi.utils import Array
from hidet.ir.type import void_p, DataType
from hidet.cuda import Stream, current_stream
from .ffi import nccl_available, NcclUniqueId

NCCL_SPLIT_NOCOLOR = -1

if nccl_available():
    from .ffi import nccl_runtime_api


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


def str_to_nccl_op(name: str) -> NcclRedOp:
    if name not in ('sum', 'prod', 'max', 'min', 'avg'):
        raise RuntimeError(f"'{name}' is not a supported reduce op")
    return getattr(NcclRedOp, name)


class NcclCommunicator:
    def __init__(self, handle: int):
        """
        Users should not call this constructor directly. Because there are two ways of creating
        a new communicator: 1) using unique_id and rank ; 2) using split.
        """
        if not nccl_available():
            raise RuntimeError("NCCL is not available")
        self._handle = handle

    def __del__(self):
        nccl_runtime_api.comm_destroy(self._handle)

    @property
    def handle(self):
        return self._handle

    def split(self, key, color):
        new_handle = nccl_runtime_api.comm_split(self._handle, color, key)
        if color == NCCL_SPLIT_NOCOLOR:
            return None
        return NcclCommunicator(new_handle)

    def all_reduce(
        self, sendbuff: int, recvbuff: int, count: int, datatype: DataType, op: str, s: Optional[Stream] = None
    ):
        if s is None:
            s = current_stream()
        nccl_runtime_api.all_reduce(
            sendbuff, recvbuff, count, int(dtype_to_nccl(datatype)), int(str_to_nccl_op(op)), self._handle, s
        )

    def broadcast(
        self, sendbuff: int, recvbuff: int, count: int, datatype: DataType, root: int, s: Optional[Stream] = None
    ):
        if s is None:
            s = current_stream()
        nccl_runtime_api.broadcast(sendbuff, recvbuff, count, int(dtype_to_nccl(datatype)), root, self._handle, s)

    def reduce(
        self,
        sendbuff: int,
        recvbuff: int,
        count: int,
        datatype: DataType,
        op: int,
        root: int,
        s: Optional[Stream] = None,
    ):
        if s is None:
            s = current_stream()
        nccl_runtime_api.reduce(
            sendbuff, recvbuff, count, int(dtype_to_nccl(datatype)), int(str_to_nccl_op(op)), root, self._handle, s
        )
    
    def all_gather(
        self, sendbuff: int, recvbuff: int, sendcount: int, datatype: DataType, s: Optional[Stream] = None,
    ):
        if s is None:
            s = current_stream()
        nccl_runtime_api.all_gather(
            sendbuff, recvbuff, sendcount, int(dtype_to_nccl(datatype)), self._handle, s
        )        
    
    def reduce_scatter(
        self, sendbuff: int, recvbuff: int, recvcount: int, datatype: DataType, op: int, s: Optional[Stream] = None
    ):
        if s is None:
            s = current_stream()
        nccl_runtime_api.reduce_scatter(
            sendbuff, recvbuff, recvcount, int(dtype_to_nccl(datatype)), int(str_to_nccl_op(op)), self._handle, s
        )



def create_comm(nranks: int, unique_id: NcclUniqueId, rank: int) -> NcclCommunicator:
    if not nccl_available():
        raise RuntimeError("NCCL is not available")
    handle = nccl_runtime_api.comm_init_rank(nranks, unique_id, rank)
    return NcclCommunicator(handle)


def comms_to_array(comms: List[NcclCommunicator]) -> Array:
    handles = [comm.handle for comm in comms]
    array = Array(void_p, len(comms))
    struct.pack_into(array.format, array.buffer, 0, *handles)
    return array


def create_unique_id() -> NcclUniqueId:
    if not nccl_available():
        raise RuntimeError("NCCL is not available")
    unique_id = NcclUniqueId()
    nccl_runtime_api.get_unique_id(unique_id)
    return unique_id


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
