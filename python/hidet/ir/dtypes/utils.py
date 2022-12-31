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
import numpy as np
from hidet.ir.type import DataType, data_type

_dtype_map = {
    'u8': np.uint8,
    'u16': np.uint16,
    'u32': np.uint32,
    'u64': np.uint64,
    'i8': np.int8,
    'i16': np.int16,
    'i32': np.int32,
    'i64': np.int64,
    'f16': np.float16,
    'bf16': None,
    'tf32': None,
    'f32': np.float32,
    'f64': np.float64,
    'bool': np.bool_,
}


def dtype_to_numpy(dtype: DataType):
    np_dtype = _dtype_map[dtype.short_name]
    if np_dtype is None:
        raise RuntimeError(f'No numpy equivalent for {dtype}')
    return np_dtype


def finfo(dtype, /):
    """
    Machine limits for integer data types.

    Parameters
    ----------
    dtype: DataType or str
        The integer data type to get the limits information.

    Returns
    -------
    ret: hidet.ir.dtypes.floats.FloatInfo
        - **bits**: *int*
          number of bits occupied by the real-valued floating-point data type.
        - **eps**: *float*
          difference between 1.0 and the next smallest representable real-valued floating-point number larger than 1.0.
        - **max**: *float*
          largest representable real-valued number.
        - **min**: *float*
          smallest representable real-valued number.
        - **smallest_normal**: *float*
          smallest positive real-valued floating-point number with full precision.
        - **dtype**: dtype
          real-valued floating-point data type.
    """
    from hidet.ir.dtypes.floats import FloatType

    if isinstance(dtype, str):
        dtype = data_type(dtype)
    if isinstance(dtype, FloatType):
        return dtype.finfo()
    else:
        raise TypeError('Expect a tensor or float data type, got {}'.format(type(dtype)))


def iinfo(dtype, /):
    """
    Machine limits for integer data types.

    Parameters
    ----------
    type: DataType or str
        The kind of integer data type about which to get information.

    Returns
    -------
    ret: hidet.ir.dtypes.integer.IntInfo
        An object having the following attributes:
        - **bits**: int
            number of bits occupied by the type.
        - **max**: int
            largest representable number.
        - **min**: int
            smallest representable number.
        - **dtype**: dtype
            integer data type.

    """
    from hidet.ir.dtypes.integer import IntegerType

    if isinstance(dtype, str):
        dtype = data_type(dtype)
    if isinstance(dtype, IntegerType):
        return dtype.iinfo()
    else:
        raise TypeError('Expect an integer tensor or data type, got {}'.format(type(dtype)))
