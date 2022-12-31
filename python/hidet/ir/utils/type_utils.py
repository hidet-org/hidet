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
from hidet.ir.type import DataType, data_type


def numeric_promotion(a_dtype: DataType, b_dtype: DataType) -> DataType:
    if a_dtype.is_float() and b_dtype.is_float():
        return a_dtype if a_dtype.nbytes >= b_dtype.nbytes else b_dtype
    elif a_dtype.is_integer() and b_dtype.is_integer():
        return a_dtype if a_dtype.nbytes >= b_dtype.nbytes else b_dtype
    elif a_dtype.is_integer() and b_dtype.is_float():
        return b_dtype
    elif a_dtype.is_float() and b_dtype.is_integer():
        return a_dtype
    else:
        raise ValueError('Cannot do numeric promotion for {} and {}.'.format(a_dtype, b_dtype))


def numeric_promotion_for_all(*dtypes: DataType) -> DataType:
    if len(dtypes) == 0:
        raise ValueError('No data type is provided.')
    elif len(dtypes) == 1:
        return dtypes[0]
    else:
        return numeric_promotion(numeric_promotion_for_all(*dtypes[:-1]), dtypes[-1])


def from_numpy_dtype(np_dtype):
    import numpy as np

    if np_dtype == np.float32:
        return data_type('float32')
    elif np_dtype == np.int32:
        return data_type('int32')
    elif np_dtype == np.int64:
        return data_type('int64')
    else:
        raise ValueError("Unrecognized numpy data type: '{}'".format(np_dtype))
