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

from hidet.ir.expr import Expr, Call
from hidet.ir.type import FuncType, VoidType, PointerType
from hidet.ir.primitives.func import register_primitive_function
from hidet.utils import initialize
from hidet.ir.primitives.func import call_primitive_func


@initialize()
def register_primitive_functions():
    functions = [
        ('avx_x86_float32x4_broadcast', '_mm_broadcast_ss', FuncType([PointerType('float32')], 'float32x4')),
        ('avx_x86_float32x4_fmadd', '_mm_fmadd_ps', FuncType(['float32x4', 'float32x4', 'float32x4'], 'float32x4')),
        ('avx_x86_float32x4_load', '_mm_loadu_ps', FuncType([PointerType('float32')], 'float32x4')),
        ('avx_x86_float32x4_store', '_mm_storeu_ps', FuncType([PointerType('float32'), 'float32x4'], VoidType())),
        ('avx_x86_float32x4_setzero', '_mm_setzero_ps', FuncType([], 'float32x4')),
        ('avx_x86_float32x8_broadcast', '_mm256_broadcast_ss', FuncType([PointerType('float32')], 'float32x8')),
        ('avx_x86_float32x8_fmadd', '_mm256_fmadd_ps', FuncType(['float32x8', 'float32x8', 'float32x8'], 'float32x8')),
        ('avx_x86_float32x8_load', '_mm256_loadu_ps', FuncType([PointerType('float32')], 'float32x8')),
        ('avx_x86_float32x8_store', '_mm256_storeu_ps', FuncType([PointerType('float32'), 'float32x8'], VoidType())),
        ('avx_x86_float32x8_setzero', '_mm256_setzero_ps', FuncType([], 'float32x8')),
        ('avx_x86_malloc', '_mm_malloc', FuncType(['uint64', 'uint64'], PointerType(VoidType()))),
        ('avx_x86_free', '_mm_free', FuncType([PointerType(VoidType())], VoidType())),
        ('x86_memset', 'memset', FuncType([PointerType(VoidType()), 'int32', 'uint64'], PointerType(VoidType()))),
        ('x86_memcpy', 'memcpy', FuncType([PointerType(VoidType()), PointerType(VoidType()), 'uint64'], PointerType(VoidType()))),
        ('aligned_alloc', 'aligned_alloc', FuncType(['int32', 'int32'], PointerType(VoidType()))),
        ('free', 'free', FuncType([PointerType(VoidType())], VoidType())),
    ]
    for name, codegen_name, func_type in functions:
        register_primitive_function(name=name, func_or_type=func_type, codegen_name=codegen_name)


def aligned_alloc(alignment: Union[int, Expr], size: Union[int, Expr]):
    return call_primitive_func('aligned_alloc', [alignment, size])


def free(addr: Expr):
    return call_primitive_func('free', [addr])


def x86_memcpy(dst: Expr, src: Expr, num: Union[Expr, int]) -> Call:
    return call_primitive_func('x86_memcpy', [dst, src, num])


def x86_memset(dst: Expr, val: Union[int, Expr], num: Union[Expr, int]) -> Call:
    return call_primitive_func('x86_memset', [dst, val, num])


def avx_malloc(size: Union[Expr, int], align: Union[Expr, int]) -> Call:
    return call_primitive_func('avx_x86_malloc', [size, align])


def avx_free(p: Expr) -> Call:
    return call_primitive_func('avx_x86_free', [p])


def avx_f32x4_setzero() -> Call:
    return call_primitive_func('avx_x86_float32x4_setzero', [])


def avx_f32x8_setzero() -> Call:
    return call_primitive_func('avx_x86_float32x8_setzero', [])


def avx_f32x4_broadcast(addr: Expr) -> Call:
    return call_primitive_func('avx_x86_float32x4_broadcast', [addr])


def avx_f32x8_broadcast(addr: Expr) -> Call:
    return call_primitive_func('avx_x86_float32x8_broadcast', [addr])


def avx_f32x4_fmadd(a: Expr, b: Expr, c: Expr) -> Call:
    return call_primitive_func('avx_x86_float32x4_fmadd', [a, b, c])


def avx_f32x8_fmadd(a: Expr, b: Expr, c: Expr) -> Call:
    return call_primitive_func('avx_x86_float32x8_fmadd', [a, b, c])


def avx_f32x4_load(addr: Expr) -> Call:
    return call_primitive_func('avx_x86_float32x4_load', [addr])


def avx_f32x8_load(addr: Expr) -> Call:
    return call_primitive_func('avx_x86_float32x8_load', [addr])


def avx_f32x4_store(addr: Expr, src: Expr) -> Call:
    return call_primitive_func('avx_x86_float32x4_store', [addr, src])


def avx_f32x8_store(addr: Expr, src: Expr) -> Call:
    return call_primitive_func('avx_x86_float32x8_store', [addr, src])

