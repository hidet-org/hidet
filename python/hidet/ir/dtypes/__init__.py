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
from .integer_subbyte import int4b, int3b, int2b, int1b, uint4b, uint3b, uint2b, uint1b
from .integer_subbyte import i4, i3, i2, i1, u4, u3, u2, u1
from .floats import float8_e4m3, float8_e5m2, float16, float32, float64, bfloat16, tfloat32
from .floats import f8e4m3, f8e5m2, f16, f32, f64, bf16, tf32
from .boolean import boolean
from .vector import (
    float16x2,
    float32x2,
    float32x4,
    float32x8,
    int8x4,
    uint8x4,
    int4bx8,
    uint4bx8,
    vectorize,
    bfloat16x2,
)
from .vector import f16x2, f32x2, f32x4, f32x8, i4x8, u4x8
from .complex import complex64, complex128
from .integer import IntegerType
from .promotion import promote_type
from .utils import dtype_to_numpy, finfo, iinfo

name2dtype = {
    'float64': float64,
    'float32': float32,
    'tfloat32': tfloat32,
    'bfloat16': bfloat16,
    'float16': float16,
    'float8_e5m2': float8_e5m2,
    'float8_e4m3': float8_e4m3,
    'int64': int64,
    'int32': int32,
    'int16': int16,
    'int8': int8,
    'uint64': uint64,
    'uint32': uint32,
    'uint16': uint16,
    'uint8': uint8,
    'bool': boolean,
    'complex64': complex64,
    'complex128': complex128,
    'float32x2': float32x2,
    'float32x4': float32x4,
    'float32x8': float32x8,
    'float16x2': float16x2,
    'int8x4': int8x4,
    'uint8x4': uint8x4,
    'int4b': int4b,
    'int3b': int3b,
    'int2b': int2b,
    'int1b': int1b,
    'uint4b': uint4b,
    'uint3b': uint3b,
    'uint2b': uint2b,
    'uint1b': uint1b,
    'int4bx8': int4bx8,
    'uint4bx8': uint4bx8,
    'bfloat16x2': bfloat16x2,
}

sname2dtype = {
    'f64': float64,
    'f32': float32,
    'tf32': tfloat32,
    'bf16': bfloat16,
    'f16': float16,
    'f8e5m2': float8_e5m2,
    'f8e4m3': float8_e4m3,
    'i64': int64,
    'i32': int32,
    'i16': int16,
    'i8': int8,
    'u64': uint64,
    'u32': uint32,
    'u16': uint16,
    'u8': uint8,
    'bool': boolean,
    'c64': complex64,
    'c128': complex128,
    'f32x2': f32x2,
    'f32x4': f32x4,
    'f32x8': f32x8,
    'f16x2': f16x2,
    'i8x4': int8x4,
    'i4': int4b,
    'i3': int3b,
    'i2': int2b,
    'i1': int1b,
    'u4': uint4b,
    'u3': uint3b,
    'u2': uint2b,
    'u1': uint1b,
    'i4x8': int4bx8,
    'u4x8': uint4bx8,
}


default_int_dtype = int32
default_index_dtype = int64
default_float_dtype = float32


def supported(name: str) -> bool:
    return name in name2dtype
