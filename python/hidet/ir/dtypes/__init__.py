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
from hidet.ir.type import DataType
from .integer import int8, int16, int32, int64, uint8, uint16, uint32, uint64
from .integer import i8, i16, i32, i64, u8, u16, u32, u64
from .floats import float16, float32, float64, bfloat16, tfloat32
from .floats import f16, f32, f64, bf16, tf32
from .boolean import boolean
from .vector import float16x2, float32x4
from .vector import f16x2, f32x4
from .promotion import promote_type
from .utils import dtype_to_numpy, finfo, iinfo

name2dtype = {
    'float64': float64,
    'float32': float32,
    'tfloat32': tfloat32,
    'bfloat16': bfloat16,
    'float16': float16,
    'int64': int64,
    'int32': int32,
    'int16': int16,
    'int8': int8,
    'uint64': uint64,
    'uint32': uint32,
    'uint16': uint16,
    'uint8': uint8,
    'bool': boolean,
    'float32x4': float32x4,
    'float16x2': float16x2,
}

sname2dtype = {
    'f64': float64,
    'f32': float32,
    'tf32': tfloat32,
    'bf16': bfloat16,
    'f16': float16,
    'i64': int64,
    'i32': int32,
    'i16': int16,
    'i8': int8,
    'u64': uint64,
    'u32': uint32,
    'u16': uint16,
    'u8': uint8,
    'bool': boolean,
    'f32x4': f32x4,
    'f16x2': f16x2,
}


default_int_dtype = int32
default_index_dtype = int64
default_float_dtype = float32


def supported(name: str) -> bool:
    return name in name2dtype
