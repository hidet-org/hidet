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
from .integer import u8, u16, u32, u64, i8, i16, i32, i64
from .floats import f16, bf16, tf32, f32, f64


_float_promotion_table = {
    (f16, f16): f16,
    (f16, bf16): bf16,
    (f16, tf32): tf32,
    (f16, f32): f32,
    (f16, f64): f64,
    (bf16, f16): bf16,
    (bf16, bf16): bf16,
    (bf16, tf32): tf32,
    (bf16, f32): f32,
    (bf16, f64): f64,
    (tf32, f16): tf32,
    (tf32, bf16): tf32,
    (tf32, tf32): tf32,
    (tf32, f32): f32,
    (tf32, f64): f64,
    (f32, f16): f32,
    (f32, bf16): f32,
    (f32, tf32): f32,
    (f32, f32): f32,
    (f32, f64): f64,
    (f64, f16): f64,
    (f64, bf16): f64,
    (f64, tf32): f64,
    (f64, f32): f64,
    (f64, f64): f64,
}

_integer_promotion_table = {
    # signed integer <op> signed integer
    (i8, i8): i8,
    (i8, i16): i16,
    (i8, i32): i32,
    (i8, i64): i64,
    (i16, i8): i16,
    (i16, i16): i16,
    (i16, i32): i32,
    (i16, i64): i64,
    (i32, i8): i32,
    (i32, i16): i32,
    (i32, i32): i32,
    (i32, i64): i64,
    (i64, i8): i64,
    (i64, i16): i64,
    (i64, i32): i64,
    (i64, i64): i64,
    # unsigned integer <op> unsigned integer
    (u8, u8): u8,
    (u8, u16): u16,
    (u8, u32): u32,
    (u8, u64): u64,
    (u16, u8): u16,
    (u16, u16): u16,
    (u16, u32): u32,
    (u16, u64): u64,
    (u32, u8): u32,
    (u32, u16): u32,
    (u32, u32): u32,
    (u32, u64): u64,
    (u64, u8): u64,
    (u64, u16): u64,
    (u64, u32): u64,
    (u64, u64): u64,
    # signed integer <op> unsigned integer
    (i8, u8): i16,
    (i8, u16): i32,
    (i8, u32): i64,
    (i8, u64): i64,
    (i16, u8): i16,
    (i16, u16): i32,
    (i16, u32): i64,
    (i16, u64): i64,
    (i32, u8): i32,
    (i32, u16): i32,
    (i32, u32): i64,
    (i32, u64): i64,
    (i64, u8): i64,
    (i64, u16): i64,
    (i64, u32): i64,
    (i64, u64): i64,
    # unsigned integer <op> signed integer
    (u8, i8): i16,
    (u8, i16): i32,
    (u8, i32): i64,
    (u8, i64): i64,
    (u16, i8): i16,
    (u16, i16): i32,
    (u16, i32): i64,
    (u16, i64): i64,
    (u32, i8): i32,
    (u32, i16): i32,
    (u32, i32): i64,
    (u32, i64): i64,
    (u64, i8): i64,
    (u64, i16): i64,
    (u64, i32): i64,
    (u64, i64): i64,
}


def promote_type(t1: DataType, t2: DataType) -> DataType:
    if t1.is_vector() or t2.is_vector():
        raise NotImplementedError("vector type promotion is not implemented")

    if t1.is_float() and t2.is_float():
        return _float_promotion_table[(t1, t2)]
    elif t1.is_float() and t2.is_integer():
        return t1
    elif t1.is_integer() and t2.is_float():
        return t2
    elif t1.is_integer() and t2.is_integer():
        return _integer_promotion_table[(t1, t2)]
    else:
        raise NotImplementedError("type promotion for {} and {} is not implemented".format(t1, t2))
