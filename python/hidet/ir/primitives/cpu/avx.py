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
        ('avx_x86_float32x4_hadd', '_mm_hadd_ps', FuncType(['float32x4', 'float32x4'], 'float32x4')),
        ('avx_x86_float32x4_fmadd', '_mm_fmadd_ps', FuncType(['float32x4', 'float32x4', 'float32x4'], 'float32x4')),
        ('avx_x86_float32x4_load', '_mm_loadu_ps', FuncType([PointerType('float32')], 'float32x4')),
        ('avx_x86_float32x4_store', '_mm_storeu_ps', FuncType([PointerType('float32'), 'float32x4'], VoidType())),
        ('avx_x86_float32x4_setzero', '_mm_setzero_ps', FuncType([], 'float32x4')),
        ('avx_x86_float32x8_set1', '_mm256_set1_ps', FuncType([PointerType('float32')], 'float32x8')),
        ('avx_x86_float32x8_broadcast', '_mm256_broadcast_ss', FuncType([PointerType('float32')], 'float32x8')),
        ('avx_x86_float32x8_fmadd', '_mm256_fmadd_ps', FuncType(['float32x8', 'float32x8', 'float32x8'], 'float32x8')),
        ('avx_x86_float32x8_load', '_mm256_loadu_ps', FuncType([PointerType('float32')], 'float32x8')),
        ('avx_x86_float32x8_store', '_mm256_storeu_ps', FuncType([PointerType('float32'), 'float32x8'], VoidType())),
        ('avx_x86_float32x8_setzero', '_mm256_setzero_ps', FuncType([], 'float32x8')),
        ('avx_x86_float32x8_add', '_mm256_add_ps', FuncType(['float32x8', 'float32x8'], 'float32x8')),
        ('avx_x86_float32x8_subtract', '_mm256_sub_ps', FuncType(['float32x8', 'float32x8'], 'float32x8')),
        ('avx_x86_float32x8_multiply', '_mm256_mul_ps', FuncType(['float32x8', 'float32x8'], 'float32x8')),
        ('avx_x86_float32x8_divide', '_mm256_div_ps', FuncType(['float32x8', 'float32x8'], 'float32x8')),
        ('avx_x86_float32x8_sqrt', '_mm256_sqrt_ps', FuncType(['float32x8'], 'float32x8')),
        ('avx_x86_float32x8_max', '_mm256_max_ps', FuncType(['float32x8', 'float32x8'], 'float32x8')),
        ('avx_x86_malloc', '_mm_malloc', FuncType(['uint64', 'uint64'], PointerType(VoidType()))),
        ('avx_x86_free', '_mm_free', FuncType([PointerType(VoidType())], VoidType())),
        ('x86_memset', 'memset', FuncType([PointerType(VoidType()), 'int32', 'uint64'], PointerType(VoidType()))),
        (
            'x86_memcpy',
            'memcpy',
            FuncType([PointerType(VoidType()), PointerType(VoidType()), 'uint64'], PointerType(VoidType())),
        ),
    ]
    for name, codegen_name, func_type in functions:
        register_primitive_function(name=name, func_or_type=func_type, codegen_name=codegen_name)

    from hidet.lang import script, attrs
    from hidet.ir.dtypes import f32x8, f32
    from hidet.ir.func import Function

    @script
    def avx_x86_f32x8_find_sum(x: f32x8) -> f32:
        attrs.func_kind = "cpu_internal"
        attrs.func_name = "avx_x86_float32x8_find_sum"
        sum_vec = call_primitive_func('avx_x86_float32x4_add',
                                      [call_primitive_func('avx_x86_float32x8_extract_half', [x, 0b0]),
                                       call_primitive_func('avx_x86_float32x8_extract_half', [x, 0b1])])
        sum_vec = call_primitive_func('avx_x86_float32x4_hadd', [sum_vec, sum_vec])
        sum_vec = call_primitive_func('avx_x86_float32x4_hadd', [sum_vec, sum_vec])
        return call_primitive_func('avx_x86_float32x4_extract_last', [sum_vec])

    assert isinstance(avx_x86_f32x8_find_sum, Function)
    register_primitive_function(avx_x86_f32x8_find_sum.name, avx_x86_f32x8_find_sum)

    @script
    def avx_x86_f32x8_find_max(x: f32x8) -> f32:
        attrs.func_kind = "cpu_internal"
        attrs.func_name = "avx_x86_float32x8_find_max"
        y = call_primitive_func('avx_x86_float32x8_permute_2f128', [x, x, 1])
        m1 = call_primitive_func('avx_x86_float32x8_max', [x, y])
        m2 = call_primitive_func('avx_x86_float32x8_permute', [m1, 0b01001110])
        m3 = call_primitive_func('avx_x86_float32x8_max', [m1, m2])
        m4 = call_primitive_func('avx_x86_float32x8_permute', [m3, 0b10110001])
        m = call_primitive_func('avx_x86_float32x8_max', [m3, m4])
        return call_primitive_func('avx_x86_float32x8_extract_last', [m])

    assert isinstance(avx_x86_f32x8_find_max, Function)
    register_primitive_function(avx_x86_f32x8_find_max.name, avx_x86_f32x8_find_max)


def avx_f32x8_find_sum(x: Expr) -> Call:
    return call_primitive_func('avx_x86_float32x8_find_sum', [x])


def avx_f32x8_find_max(x: Expr) -> Call:
    return call_primitive_func('avx_x86_float32x8_find_max', [x])


def aligned_alloc(alignment: Union[int, Expr], size: Union[int, Expr]):
    return call_primitive_func('aligned_alloc', [alignment, size])


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


def avx_f32x8_set1(a: Expr) -> Call:
    return call_primitive_func('avx_x86_float32x8_set1', [a])


def avx_f32x4_broadcast(addr: Expr) -> Call:
    return call_primitive_func('avx_x86_float32x4_broadcast', [addr])


def avx_f32x8_broadcast(addr: Expr) -> Call:
    return call_primitive_func('avx_x86_float32x8_broadcast', [addr])


def avx_f32x8_add(a: Expr, b: Expr) -> Call:
    return call_primitive_func('avx_x86_float32x8_add', [a, b])


def avx_f32x8_subtract(a: Expr, b: Expr) -> Call:
    return call_primitive_func('avx_x86_float32x8_subtract', [a, b])


def avx_f32x8_multiply(a: Expr, b: Expr) -> Call:
    return call_primitive_func('avx_x86_float32x8_multiply', [a, b])


def avx_f32x8_divide(a: Expr, b: Expr) -> Call:
    return call_primitive_func('avx_x86_float32x8_divide', [a, b])


def avx_f32x8_sqrt(a: Expr) -> Call:
    return call_primitive_func('avx_x86_float32x8_sqrt', [a])


def avx_f32x4_hadd(a: Expr, b: Expr) -> Call:
    return call_primitive_func('avx_x86_float32x4_hadd', [a, b])


def avx_f32x8_max(a: Expr, b: Expr) -> Call:
    return call_primitive_func('avx_x86_float32x8_max', [a, b])


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
