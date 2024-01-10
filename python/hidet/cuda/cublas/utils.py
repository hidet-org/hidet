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
from .ffi import cudaDataType


def as_pointer(obj) -> int:
    from hidet.graph.tensor import Tensor

    if isinstance(obj, Tensor):
        return obj.storage.addr
    elif isinstance(obj, int):
        return obj
    else:
        raise TypeError(f'Expected Tensor or int, but got {type(obj)}')


# see the definition of cudaDataType_t in <cublas_v2.h> to get the type code of each type
_type_dict = {
    dtypes.float16: cudaDataType.CUDA_R_16F,
    dtypes.float32: cudaDataType.CUDA_R_32F,
    dtypes.float64: cudaDataType.CUDA_R_64F,
}


def as_type_code(obj) -> int:
    if isinstance(obj, DataType):
        return _type_dict[obj]
    elif isinstance(obj, int):
        return obj
    else:
        raise TypeError(f'Expected DataType or int, but got {type(obj)}')
