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
from hidet.ir import dtypes
from hidet.ir.dtypes import DataType
from .ffi import cudnnDataType

_cudnn_type_dict = {
    dtypes.float16: cudnnDataType.CUDNN_DATA_HALF,
    dtypes.float32: cudnnDataType.CUDNN_DATA_FLOAT,
    dtypes.float64: cudnnDataType.CUDNN_DATA_DOUBLE,
    dtypes.int32: cudnnDataType.CUDNN_DATA_INT32,
    dtypes.int64: cudnnDataType.CUDNN_DATA_INT64,
}

_cudnn_type_dict_str = {
    "float16": cudnnDataType.CUDNN_DATA_HALF,
    "float32": cudnnDataType.CUDNN_DATA_FLOAT,
    "float64": cudnnDataType.CUDNN_DATA_DOUBLE,
    "int32": cudnnDataType.CUDNN_DATA_INT32,
    "int64": cudnnDataType.CUDNN_DATA_INT64,
}


def as_pointer(obj) -> int:
    from hidet.graph.tensor import Tensor

    if isinstance(obj, Tensor):
        return obj.storage.addr
    elif isinstance(obj, int):
        return obj
    else:
        raise TypeError(f'Expected Tensor or int, but got {type(obj)}')


def as_cudnn_type(obj) -> int:
    if isinstance(obj, DataType):
        return _cudnn_type_dict[obj]
    elif isinstance(obj, str):
        return _cudnn_type_dict_str[obj]
    elif isinstance(obj, int):
        return obj
    else:
        raise TypeError(f'Expected DataType, int, or str, but got {type(obj)}')
