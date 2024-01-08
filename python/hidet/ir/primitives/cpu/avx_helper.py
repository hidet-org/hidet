from hidet.lang import script, attrs
from hidet.ir.dtypes import f32x8, f32
from hidet.ir.func import Function
from hidet.ir.expr import Expr, Call
from hidet.ir.primitives.func import register_primitive_function, call_primitive_func
from hidet.ir.primitives.cpu.avx import (
    avx_f32x4_add,
    avx_f32x8_extract_half,
    avx_f32x4_hadd,
    avx_f32x4_extract_last,
    avx_f32x8_permute2f32x4,
    avx_f32x8_max,
    avx_f32x8_permute,
    avx_f32x8_extract_last,
)


@script
def avx_x86_f32x8_sum(x: f32x8) -> f32:
    attrs.func_kind = "cpu_internal"
    attrs.func_name = "avx_x86_float32x8_sum"
    a = avx_f32x8_extract_half(x, 0b0)
    b = avx_f32x8_extract_half(x, 0b1)
    sum_vec = avx_f32x4_add(a, b)
    sum_vec = avx_f32x4_hadd(sum_vec, sum_vec)
    sum_vec = avx_f32x4_hadd(sum_vec, sum_vec)
    return avx_f32x4_extract_last(sum_vec)


assert isinstance(avx_x86_f32x8_sum, Function)
register_primitive_function(avx_x86_f32x8_sum.name, avx_x86_f32x8_sum)


@script
def avx_x86_f32x8_scalar_max(x: f32x8) -> f32:
    attrs.func_kind = "cpu_internal"
    attrs.func_name = "avx_x86_float32x8_scalar_max"
    y = avx_f32x8_permute2f32x4(x, x, 1)
    m1 = avx_f32x8_max(x, y)
    m2 = avx_f32x8_permute(m1, 0b01001110)
    m3 = avx_f32x8_max(m1, m2)
    m4 = avx_f32x8_permute(m3, 0b10110001)
    m = avx_f32x8_max(m3, m4)
    return avx_f32x8_extract_last(m)


assert isinstance(avx_x86_f32x8_scalar_max, Function)
register_primitive_function(avx_x86_f32x8_scalar_max.name, avx_x86_f32x8_scalar_max)


def avx_f32x8_sum(x: Expr) -> Call:
    return call_primitive_func('avx_x86_float32x8_sum', [x])


def avx_f32x8_scalar_max(x: Expr) -> Call:
    return call_primitive_func('avx_x86_float32x8_scalar_max', [x])
