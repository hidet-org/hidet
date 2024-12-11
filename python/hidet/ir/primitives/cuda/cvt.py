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
from typing import Union, Optional
from hidet.utils import initialize
from hidet.ir.type import FuncType, DataType
from hidet.ir.expr import Expr, cast, deref
from hidet.ir.stmt import asm
from hidet.ir.func import Function
from hidet.ir.primitives.func import register_primitive_function, call_primitive_func, lookup_primitive_function
from hidet.ir.dtypes import vectorize
from hidet.lang import script


def resolve_cvt_func_name(src: Union[Expr, DataType], dtype: DataType) -> str:
    from hidet.ir.tools import infer_type

    if isinstance(src, DataType):
        src_dtype = src
    else:
        src_dtype = infer_type(src)
    if not isinstance(src_dtype, DataType):
        raise TypeError('src must be a scalar data type, got {}'.format(src_dtype))
    return 'cuda_cvt_{}_to_{}'.format(src_dtype.short_name, dtype.short_name)


@initialize()
def register_cvt_instructions():
    from hidet.lang import attrs
    from hidet.lang import u8, u16, u32, u64, i8, i16, i32, i64, f16, f32, f64, bf16

    for src_dtype in [u8, u16, u32, u64, i8, i16, i32, i64, bf16, f16, f32, f64]:
        for dst_dtype in [u8, u16, u32, u64, i8, i16, i32, i64, bf16, f16, f32, f64]:
            if src_dtype == dst_dtype:
                continue
            if src_dtype.is_integer() and dst_dtype.is_float():
                continue
            if src_dtype.is_float() and dst_dtype.is_integer():
                continue
            func_name = resolve_cvt_func_name(src_dtype, dst_dtype)

            @script
            def cuda_cvt(src: src_dtype) -> dst_dtype:
                attrs.func_name = func_name
                attrs.func_kind = 'cuda_internal'
                ret = dst_dtype(0)
                dst_name = dst_dtype.short_name.replace('i', 's')  # cuda use s8 to represents i8
                src_name = src_dtype.short_name.replace('i', 's')
                asm(template='cvt.{}.{} %0, %1;'.format(dst_name, src_name), outputs=[ret], inputs=[src])
                return ret

            register_primitive_function(cuda_cvt.name, cuda_cvt)


def cvt(src: Expr, dtype: DataType) -> Expr:
    """
    Convert the src expression to the given data type.

    Parameters
    ----------
    src: Expr
        The source expression to be converted.

    dtype: DataType
        The target data type.

    Returns
    -------
    ret: Expr
        The converted expression.
    """
    func_name = resolve_cvt_func_name(src, dtype)
    return call_primitive_func(func_name, args=[src])


def resolve_vectorized_cvt_func_name(
    src: Union[Expr, DataType], dst: Union[Expr, DataType], bits_per_vector: Optional[int] = None
) -> str:
    from hidet.ir.tools import infer_type
    from hidet.ir.type import PointerType, TensorType, TensorPointerType

    def get_dtype(x: Expr):
        if isinstance(x, DataType):
            dtype = x
        else:
            dtype = infer_type(x)
        if isinstance(dtype, DataType):
            return dtype
        elif isinstance(dtype, PointerType):
            return dtype.base_type
        elif isinstance(dtype, TensorType):
            return dtype.dtype
        elif isinstance(dtype, TensorPointerType):
            return dtype.tensor_type.dtype
        else:
            raise TypeError(f"failed to resolve the scalar data type for expr({x}). (got: {dtype})")

    src_dtype = get_dtype(src)
    dst_dtype = get_dtype(dst)
    if bits_per_vector is None:
        bits_per_vector = 32
    src_vector_size = bits_per_vector // src_dtype.nbits
    dst_vector_size = bits_per_vector // dst_dtype.nbits
    vector_size = max(src_vector_size, dst_vector_size)
    return f'cuda_cvt_{src_dtype.short_name}x{vector_size}_to_{dst_dtype.short_name}x{vector_size}'


@initialize()
def register_cast_u4x8_to_f16x8_interleaved():
    from hidet.lang import attrs
    from hidet.lang import u4, f16, u32
    from hidet.lang import view
    from hidet.ir.expr import right_shift

    immLut = (0xF0 & 0xCC) | 0xAA
    bottom_mask = 0x000F000F
    top_mask = 0x00F000F0
    i4s_to_f16s_magic_num = 0x64006400

    @script
    def cast_u4x8_to_f16x8_interleaved(x: u4[8], y: f16[8]):
        attrs.func_name = 'cast_u4x8_to_f16x8_interleaved'
        attrs.func_kind = 'cuda_internal'

        xi = view(x, u32[1])
        yi = view(y, u32[4])

        top_i4s = right_shift(xi[0], 8)
        asm("lop3.b32 %0, %1, %2, %3, %4;", inputs=[xi[0], bottom_mask, i4s_to_f16s_magic_num, immLut], outputs=[yi[0]])
        asm("lop3.b32 %0, %1, %2, %3, %4;", inputs=[xi[0], top_mask, i4s_to_f16s_magic_num, immLut], outputs=[yi[1]])
        asm(
            "lop3.b32 %0, %1, %2, %3, %4;",
            inputs=[top_i4s, bottom_mask, i4s_to_f16s_magic_num, immLut],
            outputs=[yi[2]],
        )
        asm("lop3.b32 %0, %1, %2, %3, %4;", inputs=[top_i4s, top_mask, i4s_to_f16s_magic_num, immLut], outputs=[yi[3]])

        f16_top_magic_num = u32(0x64006400)
        one_sixteenth = u32(0x2C002C00)
        neg_64 = u32(0xD400D400)
        asm("sub.f16x2 %0, %1, %2;", inputs=[yi[0], f16_top_magic_num], outputs=[yi[0]])
        asm("fma.rn.f16x2 %0, %1, %2, %3;", inputs=[yi[1], one_sixteenth, neg_64], outputs=[yi[1]])
        asm("sub.f16x2 %0, %1, %2;", inputs=[yi[2], f16_top_magic_num], outputs=[yi[2]])
        asm("fma.rn.f16x2 %0, %1, %2, %3;", inputs=[yi[3], one_sixteenth, neg_64], outputs=[yi[3]])

    register_primitive_function('cast_u4x8_to_f16x8_interleaved', cast_u4x8_to_f16x8_interleaved)


@initialize()
def register_vectorized_cvt_instructions():
    from hidet.lang import attrs
    from hidet.lang import u4, u2, u1, i4, i2, f16, u8, u16, u32, f32, f16x2, f32x2, i8
    from hidet.lang import view

    register_primitive_function(
        'cuda_half22float2', func_or_type=FuncType([f16x2], f32x2), codegen_name='__half22float2'
    )
    register_primitive_function(
        'cuda_float22half2', func_or_type=FuncType([f16x2], f32x2), codegen_name='__float22half2_rn'
    )

    for src_dtype in [f32, f16]:
        for dst_dtype in [f32, f16]:
            if src_dtype == dst_dtype:
                continue

            BITS_PER_GPR = 32
            src_vector_size = BITS_PER_GPR // src_dtype.nbits
            dst_vector_size = BITS_PER_GPR // dst_dtype.nbits
            vector_size = max(src_vector_size, dst_vector_size)
            func_name = resolve_vectorized_cvt_func_name(src_dtype, dst_dtype)

            src_vector_type = vectorize(src_dtype, vector_size)
            dst_vector_type = vectorize(dst_dtype, vector_size)

            if src_dtype == f32:
                primitive_func = 'cuda_float22half2'
            else:
                primitive_func = 'cuda_half22float2'

            @script
            def cuda_cvt(x: src_dtype[vector_size], y: dst_dtype[vector_size]):
                attrs.func_name = func_name
                attrs.func_kind = 'cuda_internal'

                xi = cast(x, ~src_vector_type)
                yi = cast(y, ~dst_vector_type)
                yi[0] = call_primitive_func(primitive_func, [deref(xi)])

            register_primitive_function(func_name, cuda_cvt)

    for src_dtype in [i4]:
        for dst_dtype in [i8]:
            BITS_PER_GPR = 32
            src_vector_size = BITS_PER_GPR // src_dtype.nbits
            dst_vector_size = BITS_PER_GPR // dst_dtype.nbits
            vector_size = max(src_vector_size, dst_vector_size)
            func_name = resolve_vectorized_cvt_func_name(src_dtype, dst_dtype)

            @script
            def cuda_cvt_i4x8_i8x8(x: src_dtype[vector_size], y: dst_dtype[vector_size]):
                attrs.func_name = func_name
                attrs.func_kind = 'cuda_internal'

                xi = view(x, u32[src_dtype.nbits * vector_size // 32])
                yi = view(y, u32[dst_dtype.nbits * vector_size // 32])

                asm(
                    "{ .reg .u32 tmp0, tmp1, tmp2;"
                    "shl.b32 tmp0, %2, 4;"
                    "and.b32 tmp0, tmp0, 0xf0f0f0f0;"
                    "prmt.b32 tmp1, tmp0, tmp0, 0xba98;"
                    "and.b32 tmp1, tmp1, 0xf0f0f0f0;"
                    "shr.u32 tmp0, tmp0, 4;"
                    "or.b32 tmp2, tmp0, tmp1;"
                    "and.b32 tmp0, %2, 0xf0f0f0f0;"
                    "prmt.b32 tmp1, tmp0, tmp0, 0xba98;"
                    "and.b32 tmp1, tmp1, 0xf0f0f0f0;"
                    "shr.u32 tmp0, tmp0, 4;"
                    "or.b32 tmp0, tmp0, tmp1;"
                    "prmt.b32 %0, tmp2, tmp0, 0x5140;"
                    "prmt.b32 %1, tmp2, tmp0, 0x7362;"
                    "}",
                    inputs=[xi[0]],
                    outputs=[yi[0], yi[1]],
                )

            register_primitive_function(func_name, cuda_cvt_i4x8_i8x8)

    def get_magic_numbers_4bits(input_dtype: DataType):
        if input_dtype == u4:
            immLut = (0xF0 & 0xCC) | 0xAA
            xor_mask = 0x64006400
            and_mask = 0xFFF0FF0F
            add = u32(0xD400E400)
            mul = u32(0x2C003C00)
            return immLut, xor_mask, and_mask, mul, add
        else:
            assert input_dtype == i4
            immLut = (0xF0 & 0xCC) ^ 0xAA
            xor_mask = 0x64806408
            and_mask = 0xFFF0FF0F
            add = u32(0xD480E408)
            mul = u32(0x2C003C00)
            return immLut, xor_mask, and_mask, mul, add

    for src_dtype in [i4, u4]:
        for dst_dtype in [f16]:
            for bits_per_vector in [32, 16, 8]:
                src_vector_size = bits_per_vector // src_dtype.nbits
                dst_vector_size = bits_per_vector // dst_dtype.nbits
                vector_size = max(src_vector_size, dst_vector_size)
                func_name = resolve_vectorized_cvt_func_name(src_dtype, dst_dtype, bits_per_vector)

                immLut, xor_mask, and_mask, mul, add = get_magic_numbers_4bits(src_dtype)

                if bits_per_vector == 32:

                    @script
                    def cuda_cvt_u4x8_f16x8(x: src_dtype[vector_size], y: dst_dtype[vector_size]):
                        attrs.func_name = func_name
                        attrs.func_kind = 'cuda_internal'

                        xi = view(x, u32[src_dtype.nbits * vector_size // 32])
                        yi = view(y, u32[dst_dtype.nbits * vector_size // 32])

                        asm("prmt.b32 %0, %1, %2, %3;", inputs=[xi[0], u32(0), u32(0x4040)], outputs=[yi[0]])
                        asm("prmt.b32 %0, %1, %2, %3;", inputs=[xi[0], u32(0), u32(0x4141)], outputs=[yi[1]])
                        asm("prmt.b32 %0, %1, %2, %3;", inputs=[xi[0], u32(0), u32(0x4242)], outputs=[yi[2]])
                        asm("prmt.b32 %0, %1, %2, %3;", inputs=[xi[0], u32(0), u32(0x4343)], outputs=[yi[3]])

                        asm("lop3.b32 %0, %1, %2, %3, %4;", inputs=[yi[0], and_mask, xor_mask, immLut], outputs=[yi[0]])
                        asm("lop3.b32 %0, %1, %2, %3, %4;", inputs=[yi[1], and_mask, xor_mask, immLut], outputs=[yi[1]])
                        asm("lop3.b32 %0, %1, %2, %3, %4;", inputs=[yi[2], and_mask, xor_mask, immLut], outputs=[yi[2]])
                        asm("lop3.b32 %0, %1, %2, %3, %4;", inputs=[yi[3], and_mask, xor_mask, immLut], outputs=[yi[3]])

                        hfma_scale_rep = u32(mul)
                        hfma_bias_rep = u32(add)
                        asm(
                            "fma.rn.f16x2 %0, %1, %2, %3;",
                            inputs=[hfma_scale_rep, yi[0], hfma_bias_rep],
                            outputs=[yi[0]],
                        )
                        asm(
                            "fma.rn.f16x2 %0, %1, %2, %3;",
                            inputs=[hfma_scale_rep, yi[1], hfma_bias_rep],
                            outputs=[yi[1]],
                        )
                        asm(
                            "fma.rn.f16x2 %0, %1, %2, %3;",
                            inputs=[hfma_scale_rep, yi[2], hfma_bias_rep],
                            outputs=[yi[2]],
                        )
                        asm(
                            "fma.rn.f16x2 %0, %1, %2, %3;",
                            inputs=[hfma_scale_rep, yi[3], hfma_bias_rep],
                            outputs=[yi[3]],
                        )

                    register_primitive_function(func_name, cuda_cvt_u4x8_f16x8)
                elif bits_per_vector == 16:

                    @script
                    def cuda_cvt_u4x4_f16x4(x: src_dtype[vector_size], y: dst_dtype[vector_size]):
                        attrs.func_name = func_name
                        attrs.func_kind = 'cuda_internal'

                        x16 = view(x, u16[1])
                        xi = cvt(x16[0], u32)
                        yi = view(y, u32[dst_dtype.nbits * vector_size // 32])

                        asm("prmt.b32 %0, %1, %2, %3;", inputs=[xi, u32(0), u32(0x4040)], outputs=[yi[0]])
                        asm("prmt.b32 %0, %1, %2, %3;", inputs=[xi, u32(0), u32(0x4141)], outputs=[yi[1]])

                        asm("lop3.b32 %0, %1, %2, %3, %4;", inputs=[yi[0], and_mask, xor_mask, immLut], outputs=[yi[0]])
                        asm("lop3.b32 %0, %1, %2, %3, %4;", inputs=[yi[1], and_mask, xor_mask, immLut], outputs=[yi[1]])

                        hfma_scale_rep = u32(mul)
                        hfma_bias_rep = u32(add)
                        asm(
                            "fma.rn.f16x2 %0, %1, %2, %3;",
                            inputs=[hfma_scale_rep, yi[0], hfma_bias_rep],
                            outputs=[yi[0]],
                        )
                        asm(
                            "fma.rn.f16x2 %0, %1, %2, %3;",
                            inputs=[hfma_scale_rep, yi[1], hfma_bias_rep],
                            outputs=[yi[1]],
                        )

                    register_primitive_function(func_name, cuda_cvt_u4x4_f16x4)
                elif bits_per_vector == 8:

                    @script
                    def cuda_cvt_u4x2_f16x2(x: src_dtype[vector_size], y: dst_dtype[vector_size]):
                        attrs.func_name = func_name
                        attrs.func_kind = 'cuda_internal'

                        x8 = view(x, u8[1])
                        xi = cast(x8[0], u32)
                        yi = view(y, u32[dst_dtype.nbits * vector_size // 32])

                        asm("prmt.b32 %0, %1, %2, %3;", inputs=[xi, u32(0), u32(0x4040)], outputs=[yi[0]])

                        asm("lop3.b32 %0, %1, %2, %3, %4;", inputs=[yi[0], and_mask, xor_mask, immLut], outputs=[yi[0]])

                        hfma_scale_rep = u32(mul)
                        hfma_bias_rep = u32(add)
                        asm(
                            "fma.rn.f16x2 %0, %1, %2, %3;",
                            inputs=[hfma_scale_rep, yi[0], hfma_bias_rep],
                            outputs=[yi[0]],
                        )

                    register_primitive_function(func_name, cuda_cvt_u4x2_f16x2)

    def get_magic_numbers_2bits(input_dtype: DataType):
        if input_dtype == u2:
            immLut = (0xF0 & 0xCC) | 0xAA
            xor_mask = 0x64006400
            lo_and_mask = 0xFF0CFF03
            hi_and_mask = 0xFFC0FF30
            add_lo = u32(0xDC00E400)
            mul_lo = u32(0x34003C00)
            add_hi = u32(0xCC00D400)
            mul_hi = u32(0x24002C00)
            return immLut, xor_mask, xor_mask, lo_and_mask, hi_and_mask, mul_lo, add_lo, mul_hi, add_hi
        else:
            assert input_dtype == i2
            immLut = (0xF0 & 0xCC) ^ 0xAA
            lo_xor_mask = 0x64086402
            hi_xor_mask = 0x64806420
            lo_and_mask = 0xFF0CFF03
            hi_and_mask = 0xFFC0FF30
            add_lo = u32(0xDC08E402)
            mul_lo = u32(0x34003C00)
            add_hi = u32(0xCC80D420)
            mul_hi = u32(0x24002C00)
            return immLut, lo_xor_mask, hi_xor_mask, lo_and_mask, hi_and_mask, mul_lo, add_lo, mul_hi, add_hi

    for src_dtype in [i2, u2]:
        for dst_dtype in [f16]:
            for bits_per_vector in [32, 16, 8]:
                src_vector_size = bits_per_vector // src_dtype.nbits
                dst_vector_size = bits_per_vector // dst_dtype.nbits
                vector_size = max(src_vector_size, dst_vector_size)
                func_name = resolve_vectorized_cvt_func_name(src_dtype, dst_dtype, bits_per_vector)

                (
                    immLut,
                    lo_xor_mask,
                    hi_xor_mask,
                    lo_and_mask,
                    hi_and_mask,
                    mul_lo,
                    add_lo,
                    mul_hi,
                    add_hi,
                ) = get_magic_numbers_2bits(src_dtype)

                if bits_per_vector == 32:

                    @script
                    def cuda_cvt_u2x16_f16x16(x: src_dtype[vector_size], y: dst_dtype[vector_size]):
                        attrs.func_name = func_name
                        attrs.func_kind = 'cuda_internal'

                        xi = view(x, u32[src_dtype.nbits * vector_size // 32])
                        yi = view(y, u32[dst_dtype.nbits * vector_size // 32])

                        # xi ~ 16 2-bit integers
                        # 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0
                        # y0 ~ 0, 0, 0, ..., 0, 1, 2, 3, 0, 0, 0, ..., 0, 1, 2, 3
                        asm("prmt.b32 %0, %1, %2, %3;", inputs=[xi[0], 0, 0x4040], outputs=[yi[0]])
                        asm(
                            "lop3.b32 %0, %1, %2, %3, %4;",
                            inputs=[yi[0], hi_and_mask, hi_xor_mask, immLut],
                            outputs=[yi[1]],
                        )
                        asm("prmt.b32 %0, %1, %2, %3;", inputs=[xi[0], 0, 0x4141], outputs=[yi[2]])
                        asm(
                            "lop3.b32 %0, %1, %2, %3, %4;",
                            inputs=[yi[2], hi_and_mask, hi_xor_mask, immLut],
                            outputs=[yi[3]],
                        )
                        asm("prmt.b32 %0, %1, %2, %3;", inputs=[xi[0], 0, 0x4242], outputs=[yi[4]])
                        asm(
                            "lop3.b32 %0, %1, %2, %3, %4;",
                            inputs=[yi[4], hi_and_mask, hi_xor_mask, immLut],
                            outputs=[yi[5]],
                        )
                        asm("prmt.b32 %0, %1, %2, %3;", inputs=[xi[0], 0, 0x4343], outputs=[yi[6]])
                        asm(
                            "lop3.b32 %0, %1, %2, %3, %4;",
                            inputs=[yi[6], hi_and_mask, hi_xor_mask, immLut],
                            outputs=[yi[7]],
                        )

                        asm(
                            "lop3.b32 %0, %1, %2, %3, %4;",
                            inputs=[yi[0], lo_and_mask, lo_xor_mask, immLut],
                            outputs=[yi[0]],
                        )
                        asm(
                            "lop3.b32 %0, %1, %2, %3, %4;",
                            inputs=[yi[2], lo_and_mask, lo_xor_mask, immLut],
                            outputs=[yi[2]],
                        )
                        asm(
                            "lop3.b32 %0, %1, %2, %3, %4;",
                            inputs=[yi[4], lo_and_mask, lo_xor_mask, immLut],
                            outputs=[yi[4]],
                        )
                        asm(
                            "lop3.b32 %0, %1, %2, %3, %4;",
                            inputs=[yi[6], lo_and_mask, lo_xor_mask, immLut],
                            outputs=[yi[6]],
                        )

                        lo_scale_rep = u32(mul_lo)
                        lo_bias_rep = u32(add_lo)
                        hi_scale_rep = u32(mul_hi)
                        hi_bias_rep = u32(add_hi)
                        asm("fma.rn.f16x2 %0, %1, %2, %3;", inputs=[lo_scale_rep, yi[0], lo_bias_rep], outputs=[yi[0]])
                        asm("fma.rn.f16x2 %0, %1, %2, %3;", inputs=[lo_scale_rep, yi[2], lo_bias_rep], outputs=[yi[2]])
                        asm("fma.rn.f16x2 %0, %1, %2, %3;", inputs=[lo_scale_rep, yi[4], lo_bias_rep], outputs=[yi[4]])
                        asm("fma.rn.f16x2 %0, %1, %2, %3;", inputs=[lo_scale_rep, yi[6], lo_bias_rep], outputs=[yi[6]])
                        asm("fma.rn.f16x2 %0, %1, %2, %3;", inputs=[hi_scale_rep, yi[1], hi_bias_rep], outputs=[yi[1]])
                        asm("fma.rn.f16x2 %0, %1, %2, %3;", inputs=[hi_scale_rep, yi[3], hi_bias_rep], outputs=[yi[3]])
                        asm("fma.rn.f16x2 %0, %1, %2, %3;", inputs=[hi_scale_rep, yi[5], hi_bias_rep], outputs=[yi[5]])
                        asm("fma.rn.f16x2 %0, %1, %2, %3;", inputs=[hi_scale_rep, yi[7], hi_bias_rep], outputs=[yi[7]])

                    register_primitive_function(func_name, cuda_cvt_u2x16_f16x16)
                elif bits_per_vector == 16:

                    @script
                    def cuda_cvt_u2x8_f16x8(x: src_dtype[vector_size], y: dst_dtype[vector_size]):
                        attrs.func_name = func_name
                        attrs.func_kind = 'cuda_internal'

                        x16 = view(x, u16[1])
                        xi = cvt(x16[0], u32)
                        yi = view(y, u32[dst_dtype.nbits * vector_size // 32])

                        # xi ~ 16 2-bit integers
                        # 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0
                        # y0 ~ 0, 0, 0, ..., 0, 1, 2, 3, 0, 0, 0, ..., 0, 1, 2, 3
                        asm("prmt.b32 %0, %1, %2, %3;", inputs=[xi, 0, 0x4040], outputs=[yi[0]])
                        asm(
                            "lop3.b32 %0, %1, %2, %3, %4;",
                            inputs=[yi[0], hi_and_mask, hi_xor_mask, immLut],
                            outputs=[yi[1]],
                        )
                        asm("prmt.b32 %0, %1, %2, %3;", inputs=[xi, 0, 0x4141], outputs=[yi[2]])
                        asm(
                            "lop3.b32 %0, %1, %2, %3, %4;",
                            inputs=[yi[2], hi_and_mask, hi_xor_mask, immLut],
                            outputs=[yi[3]],
                        )

                        asm(
                            "lop3.b32 %0, %1, %2, %3, %4;",
                            inputs=[yi[0], lo_and_mask, lo_xor_mask, immLut],
                            outputs=[yi[0]],
                        )
                        asm(
                            "lop3.b32 %0, %1, %2, %3, %4;",
                            inputs=[yi[2], lo_and_mask, lo_xor_mask, immLut],
                            outputs=[yi[2]],
                        )

                        lo_scale_rep = u32(mul_lo)
                        lo_bias_rep = u32(add_lo)
                        hi_scale_rep = u32(mul_hi)
                        hi_bias_rep = u32(add_hi)
                        asm("fma.rn.f16x2 %0, %1, %2, %3;", inputs=[lo_scale_rep, yi[0], lo_bias_rep], outputs=[yi[0]])
                        asm("fma.rn.f16x2 %0, %1, %2, %3;", inputs=[lo_scale_rep, yi[2], lo_bias_rep], outputs=[yi[2]])
                        asm("fma.rn.f16x2 %0, %1, %2, %3;", inputs=[hi_scale_rep, yi[1], hi_bias_rep], outputs=[yi[1]])
                        asm("fma.rn.f16x2 %0, %1, %2, %3;", inputs=[hi_scale_rep, yi[3], hi_bias_rep], outputs=[yi[3]])

                    register_primitive_function(func_name, cuda_cvt_u2x8_f16x8)
                elif bits_per_vector == 8:
                    src_vector_size = bits_per_vector // src_dtype.nbits
                    dst_vector_size = bits_per_vector // dst_dtype.nbits
                    vector_size = max(src_vector_size, dst_vector_size)
                    func_name = resolve_vectorized_cvt_func_name(src_dtype, dst_dtype, bits_per_vector)

                    immLut = (0xF0 & 0xCC) | 0xAA
                    xor_mask = 0x64006400
                    lo_and_mask = 0xFF0CFF03
                    hi_and_mask = 0xFFC0FF30

                    @script
                    def cuda_cvt_u2x4_f16x4(x: src_dtype[vector_size], y: dst_dtype[vector_size]):
                        attrs.func_name = func_name
                        attrs.func_kind = 'cuda_internal'

                        x8 = view(x, u8[1])
                        xi = cast(x8[0], u32)
                        yi = view(y, u32[dst_dtype.nbits * vector_size // 32])

                        # xi ~ 16 2-bit integers
                        # 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0
                        # y0 ~ 0, 0, 0, ..., 0, 1, 2, 3, 0, 0, 0, ..., 0, 1, 2, 3
                        asm("prmt.b32 %0, %1, %2, %3;", inputs=[xi, 0, 0x4040], outputs=[yi[0]])
                        asm(
                            "lop3.b32 %0, %1, %2, %3, %4;",
                            inputs=[yi[0], hi_and_mask, hi_xor_mask, immLut],
                            outputs=[yi[1]],
                        )
                        asm(
                            "lop3.b32 %0, %1, %2, %3, %4;",
                            inputs=[yi[0], lo_and_mask, lo_xor_mask, immLut],
                            outputs=[yi[0]],
                        )

                        lo_scale_rep = u32(mul_lo)
                        lo_bias_rep = u32(add_lo)
                        hi_scale_rep = u32(mul_hi)
                        hi_bias_rep = u32(add_hi)
                        asm("fma.rn.f16x2 %0, %1, %2, %3;", inputs=[lo_scale_rep, yi[0], lo_bias_rep], outputs=[yi[0]])
                        asm("fma.rn.f16x2 %0, %1, %2, %3;", inputs=[hi_scale_rep, yi[1], hi_bias_rep], outputs=[yi[1]])

                    register_primitive_function(func_name, cuda_cvt_u2x4_f16x4)

    for src_dtype in [u1]:
        for dst_dtype in [f16]:
            for bits_per_vector in [32, 16, 8]:
                src_vector_size = bits_per_vector // src_dtype.nbits
                dst_vector_size = bits_per_vector // dst_dtype.nbits
                vector_size = max(src_vector_size, dst_vector_size)
                func_name = resolve_vectorized_cvt_func_name(src_dtype, dst_dtype, bits_per_vector)

                immLut = (0xF0 & 0xCC) | 0xAA
                xor_mask = 0x64006400
                quad1_and_mask = 0xFF02FF01
                quad2_and_mask = 0xFF08FF04
                quad3_and_mask = 0xFF20FF10
                quad4_and_mask = 0xFF80FF40
                quad1_mul = u32(0x38003C00)
                quad1_add = u32(0xE000E400)
                quad2_mul = u32(0x30003400)
                quad2_add = u32(0xD800DC00)
                quad3_mul = u32(0x28002C00)
                quad3_add = u32(0xD000D400)
                quad4_mul = u32(0x20002400)
                quad4_add = u32(0xC800CC00)

                if bits_per_vector == 32:

                    @script
                    def cuda_cvt_u1x32_f16x32(x: src_dtype[vector_size], y: dst_dtype[vector_size]):
                        attrs.func_name = func_name
                        attrs.func_kind = 'cuda_internal'

                        xi = view(x, u32[src_dtype.nbits * vector_size // 32])
                        yi = view(y, u32[dst_dtype.nbits * vector_size // 32])

                        asm("prmt.b32 %0, %1, %2, %3;", inputs=[xi[0], 0, 0x4040], outputs=[yi[0]])
                        asm(
                            "lop3.b32 %0, %1, %2, %3, %4;",
                            inputs=[yi[0], quad2_and_mask, xor_mask, immLut],
                            outputs=[yi[1]],
                        )
                        asm(
                            "lop3.b32 %0, %1, %2, %3, %4;",
                            inputs=[yi[0], quad3_and_mask, xor_mask, immLut],
                            outputs=[yi[2]],
                        )
                        asm(
                            "lop3.b32 %0, %1, %2, %3, %4;",
                            inputs=[yi[0], quad4_and_mask, xor_mask, immLut],
                            outputs=[yi[3]],
                        )

                        asm("prmt.b32 %0, %1, %2, %3;", inputs=[xi[0], 0, 0x4141], outputs=[yi[4]])
                        asm(
                            "lop3.b32 %0, %1, %2, %3, %4;",
                            inputs=[yi[4], quad2_and_mask, xor_mask, immLut],
                            outputs=[yi[5]],
                        )
                        asm(
                            "lop3.b32 %0, %1, %2, %3, %4;",
                            inputs=[yi[4], quad3_and_mask, xor_mask, immLut],
                            outputs=[yi[6]],
                        )
                        asm(
                            "lop3.b32 %0, %1, %2, %3, %4;",
                            inputs=[yi[4], quad4_and_mask, xor_mask, immLut],
                            outputs=[yi[7]],
                        )

                        asm("prmt.b32 %0, %1, %2, %3;", inputs=[xi[0], 0, 0x4242], outputs=[yi[8]])
                        asm(
                            "lop3.b32 %0, %1, %2, %3, %4;",
                            inputs=[yi[8], quad2_and_mask, xor_mask, immLut],
                            outputs=[yi[9]],
                        )
                        asm(
                            "lop3.b32 %0, %1, %2, %3, %4;",
                            inputs=[yi[8], quad3_and_mask, xor_mask, immLut],
                            outputs=[yi[10]],
                        )
                        asm(
                            "lop3.b32 %0, %1, %2, %3, %4;",
                            inputs=[yi[8], quad4_and_mask, xor_mask, immLut],
                            outputs=[yi[11]],
                        )

                        asm("prmt.b32 %0, %1, %2, %3;", inputs=[xi[0], 0, 0x4343], outputs=[yi[12]])
                        asm(
                            "lop3.b32 %0, %1, %2, %3, %4;",
                            inputs=[yi[12], quad2_and_mask, xor_mask, immLut],
                            outputs=[yi[13]],
                        )
                        asm(
                            "lop3.b32 %0, %1, %2, %3, %4;",
                            inputs=[yi[12], quad3_and_mask, xor_mask, immLut],
                            outputs=[yi[14]],
                        )
                        asm(
                            "lop3.b32 %0, %1, %2, %3, %4;",
                            inputs=[yi[12], quad4_and_mask, xor_mask, immLut],
                            outputs=[yi[15]],
                        )

                        asm(
                            "lop3.b32 %0, %1, %2, %3, %4;",
                            inputs=[yi[0], quad1_and_mask, xor_mask, immLut],
                            outputs=[yi[0]],
                        )
                        asm(
                            "lop3.b32 %0, %1, %2, %3, %4;",
                            inputs=[yi[4], quad1_and_mask, xor_mask, immLut],
                            outputs=[yi[4]],
                        )
                        asm(
                            "lop3.b32 %0, %1, %2, %3, %4;",
                            inputs=[yi[8], quad1_and_mask, xor_mask, immLut],
                            outputs=[yi[8]],
                        )
                        asm(
                            "lop3.b32 %0, %1, %2, %3, %4;",
                            inputs=[yi[12], quad1_and_mask, xor_mask, immLut],
                            outputs=[yi[12]],
                        )

                        quad1_scale_rep = u32(quad1_mul)
                        quad1_bias_rep = u32(quad1_add)
                        quad2_scale_rep = u32(quad2_mul)
                        quad2_bias_rep = u32(quad2_add)
                        quad3_scale_rep = u32(quad3_mul)
                        quad3_bias_rep = u32(quad3_add)
                        quad4_scale_rep = u32(quad4_mul)
                        quad4_bias_rep = u32(quad4_add)

                        asm(
                            "fma.rn.f16x2 %0, %1, %2, %3;",
                            inputs=[quad1_scale_rep, yi[0], quad1_bias_rep],
                            outputs=[yi[0]],
                        )
                        asm(
                            "fma.rn.f16x2 %0, %1, %2, %3;",
                            inputs=[quad1_scale_rep, yi[4], quad1_bias_rep],
                            outputs=[yi[4]],
                        )
                        asm(
                            "fma.rn.f16x2 %0, %1, %2, %3;",
                            inputs=[quad1_scale_rep, yi[8], quad1_bias_rep],
                            outputs=[yi[8]],
                        )
                        asm(
                            "fma.rn.f16x2 %0, %1, %2, %3;",
                            inputs=[quad1_scale_rep, yi[12], quad1_bias_rep],
                            outputs=[yi[12]],
                        )

                        asm(
                            "fma.rn.f16x2 %0, %1, %2, %3;",
                            inputs=[quad2_scale_rep, yi[1], quad2_bias_rep],
                            outputs=[yi[1]],
                        )
                        asm(
                            "fma.rn.f16x2 %0, %1, %2, %3;",
                            inputs=[quad2_scale_rep, yi[5], quad2_bias_rep],
                            outputs=[yi[5]],
                        )
                        asm(
                            "fma.rn.f16x2 %0, %1, %2, %3;",
                            inputs=[quad2_scale_rep, yi[9], quad2_bias_rep],
                            outputs=[yi[9]],
                        )
                        asm(
                            "fma.rn.f16x2 %0, %1, %2, %3;",
                            inputs=[quad2_scale_rep, yi[13], quad2_bias_rep],
                            outputs=[yi[13]],
                        )

                        asm(
                            "fma.rn.f16x2 %0, %1, %2, %3;",
                            inputs=[quad3_scale_rep, yi[2], quad3_bias_rep],
                            outputs=[yi[2]],
                        )
                        asm(
                            "fma.rn.f16x2 %0, %1, %2, %3;",
                            inputs=[quad3_scale_rep, yi[6], quad3_bias_rep],
                            outputs=[yi[6]],
                        )
                        asm(
                            "fma.rn.f16x2 %0, %1, %2, %3;",
                            inputs=[quad3_scale_rep, yi[10], quad3_bias_rep],
                            outputs=[yi[10]],
                        )
                        asm(
                            "fma.rn.f16x2 %0, %1, %2, %3;",
                            inputs=[quad3_scale_rep, yi[14], quad3_bias_rep],
                            outputs=[yi[14]],
                        )

                        asm(
                            "fma.rn.f16x2 %0, %1, %2, %3;",
                            inputs=[quad4_scale_rep, yi[3], quad4_bias_rep],
                            outputs=[yi[3]],
                        )
                        asm(
                            "fma.rn.f16x2 %0, %1, %2, %3;",
                            inputs=[quad4_scale_rep, yi[7], quad4_bias_rep],
                            outputs=[yi[7]],
                        )
                        asm(
                            "fma.rn.f16x2 %0, %1, %2, %3;",
                            inputs=[quad4_scale_rep, yi[11], quad4_bias_rep],
                            outputs=[yi[11]],
                        )
                        asm(
                            "fma.rn.f16x2 %0, %1, %2, %3;",
                            inputs=[quad4_scale_rep, yi[15], quad4_bias_rep],
                            outputs=[yi[15]],
                        )

                    register_primitive_function(func_name, cuda_cvt_u1x32_f16x32)
                elif bits_per_vector == 16:

                    @script
                    def cuda_cvt_u1x16_f16x16(x: src_dtype[vector_size], y: dst_dtype[vector_size]):
                        attrs.func_name = func_name
                        attrs.func_kind = 'cuda_internal'

                        x16 = view(x, u16[1])
                        xi = cvt(x16[0], u32)
                        yi = view(y, u32[dst_dtype.nbits * vector_size // 32])

                        asm("prmt.b32 %0, %1, %2, %3;", inputs=[xi, 0, 0x4040], outputs=[yi[0]])
                        asm(
                            "lop3.b32 %0, %1, %2, %3, %4;",
                            inputs=[yi[0], quad2_and_mask, xor_mask, immLut],
                            outputs=[yi[1]],
                        )
                        asm(
                            "lop3.b32 %0, %1, %2, %3, %4;",
                            inputs=[yi[0], quad3_and_mask, xor_mask, immLut],
                            outputs=[yi[2]],
                        )
                        asm(
                            "lop3.b32 %0, %1, %2, %3, %4;",
                            inputs=[yi[0], quad4_and_mask, xor_mask, immLut],
                            outputs=[yi[3]],
                        )

                        asm("prmt.b32 %0, %1, %2, %3;", inputs=[xi, 0, 0x4141], outputs=[yi[4]])
                        asm(
                            "lop3.b32 %0, %1, %2, %3, %4;",
                            inputs=[yi[4], quad2_and_mask, xor_mask, immLut],
                            outputs=[yi[5]],
                        )
                        asm(
                            "lop3.b32 %0, %1, %2, %3, %4;",
                            inputs=[yi[4], quad3_and_mask, xor_mask, immLut],
                            outputs=[yi[6]],
                        )
                        asm(
                            "lop3.b32 %0, %1, %2, %3, %4;",
                            inputs=[yi[4], quad4_and_mask, xor_mask, immLut],
                            outputs=[yi[7]],
                        )

                        asm(
                            "lop3.b32 %0, %1, %2, %3, %4;",
                            inputs=[yi[0], quad1_and_mask, xor_mask, immLut],
                            outputs=[yi[0]],
                        )
                        asm(
                            "lop3.b32 %0, %1, %2, %3, %4;",
                            inputs=[yi[4], quad1_and_mask, xor_mask, immLut],
                            outputs=[yi[4]],
                        )

                        quad1_scale_rep = u32(quad1_mul)
                        quad1_bias_rep = u32(quad1_add)
                        quad2_scale_rep = u32(quad2_mul)
                        quad2_bias_rep = u32(quad2_add)
                        quad3_scale_rep = u32(quad3_mul)
                        quad3_bias_rep = u32(quad3_add)
                        quad4_scale_rep = u32(quad4_mul)
                        quad4_bias_rep = u32(quad4_add)

                        asm(
                            "fma.rn.f16x2 %0, %1, %2, %3;",
                            inputs=[quad1_scale_rep, yi[0], quad1_bias_rep],
                            outputs=[yi[0]],
                        )
                        asm(
                            "fma.rn.f16x2 %0, %1, %2, %3;",
                            inputs=[quad1_scale_rep, yi[4], quad1_bias_rep],
                            outputs=[yi[4]],
                        )

                        asm(
                            "fma.rn.f16x2 %0, %1, %2, %3;",
                            inputs=[quad2_scale_rep, yi[1], quad2_bias_rep],
                            outputs=[yi[1]],
                        )
                        asm(
                            "fma.rn.f16x2 %0, %1, %2, %3;",
                            inputs=[quad2_scale_rep, yi[5], quad2_bias_rep],
                            outputs=[yi[5]],
                        )

                        asm(
                            "fma.rn.f16x2 %0, %1, %2, %3;",
                            inputs=[quad3_scale_rep, yi[2], quad3_bias_rep],
                            outputs=[yi[2]],
                        )
                        asm(
                            "fma.rn.f16x2 %0, %1, %2, %3;",
                            inputs=[quad3_scale_rep, yi[6], quad3_bias_rep],
                            outputs=[yi[6]],
                        )

                        asm(
                            "fma.rn.f16x2 %0, %1, %2, %3;",
                            inputs=[quad4_scale_rep, yi[3], quad4_bias_rep],
                            outputs=[yi[3]],
                        )
                        asm(
                            "fma.rn.f16x2 %0, %1, %2, %3;",
                            inputs=[quad4_scale_rep, yi[7], quad4_bias_rep],
                            outputs=[yi[7]],
                        )

                    register_primitive_function(func_name, cuda_cvt_u1x16_f16x16)
                elif bits_per_vector == 8:

                    @script
                    def cuda_cvt_u1x8_f16x8(x: src_dtype[vector_size], y: dst_dtype[vector_size]):
                        attrs.func_name = func_name
                        attrs.func_kind = 'cuda_internal'

                        x8 = view(x, u8[1])
                        xi = cast(x8[0], u32)
                        yi = view(y, u32[dst_dtype.nbits * vector_size // 32])

                        asm("prmt.b32 %0, %1, %2, %3;", inputs=[xi, 0, 0x4040], outputs=[yi[0]])
                        asm(
                            "lop3.b32 %0, %1, %2, %3, %4;",
                            inputs=[yi[0], quad2_and_mask, xor_mask, immLut],
                            outputs=[yi[1]],
                        )
                        asm(
                            "lop3.b32 %0, %1, %2, %3, %4;",
                            inputs=[yi[0], quad3_and_mask, xor_mask, immLut],
                            outputs=[yi[2]],
                        )
                        asm(
                            "lop3.b32 %0, %1, %2, %3, %4;",
                            inputs=[yi[0], quad4_and_mask, xor_mask, immLut],
                            outputs=[yi[3]],
                        )
                        asm(
                            "lop3.b32 %0, %1, %2, %3, %4;",
                            inputs=[yi[0], quad1_and_mask, xor_mask, immLut],
                            outputs=[yi[0]],
                        )

                        quad1_scale_rep = u32(quad1_mul)
                        quad1_bias_rep = u32(quad1_add)
                        quad2_scale_rep = u32(quad2_mul)
                        quad2_bias_rep = u32(quad2_add)
                        quad3_scale_rep = u32(quad3_mul)
                        quad3_bias_rep = u32(quad3_add)
                        quad4_scale_rep = u32(quad4_mul)
                        quad4_bias_rep = u32(quad4_add)

                        asm(
                            "fma.rn.f16x2 %0, %1, %2, %3;",
                            inputs=[quad1_scale_rep, yi[0], quad1_bias_rep],
                            outputs=[yi[0]],
                        )
                        asm(
                            "fma.rn.f16x2 %0, %1, %2, %3;",
                            inputs=[quad2_scale_rep, yi[1], quad2_bias_rep],
                            outputs=[yi[1]],
                        )
                        asm(
                            "fma.rn.f16x2 %0, %1, %2, %3;",
                            inputs=[quad3_scale_rep, yi[2], quad3_bias_rep],
                            outputs=[yi[2]],
                        )
                        asm(
                            "fma.rn.f16x2 %0, %1, %2, %3;",
                            inputs=[quad4_scale_rep, yi[3], quad4_bias_rep],
                            outputs=[yi[3]],
                        )

                    register_primitive_function(func_name, cuda_cvt_u1x8_f16x8)


def cvtv(src: Expr, dst: Expr, elements_per_vector: Optional[int] = None):
    """
    Convert the src vector to the given data type.

    Parameters
    ----------
    src: Expr
        The source expression to be converted.

        Note:
        1. The argument, src, can be either a pointer to the buffer to be converted, a register tensor, or a vector
        2. The size of the buffer that holds src should match the given vector size.

    dst: Expr
        The destination expression.

        Note:
        1. The argument, dst, can be either a pointer to the buffer to be converted, a register tensor, or a vector
        2. The size of the buffer that holds dst should match the given vector size.

    elements_per_vector:
        The number of elements in a vector.
        By default, the vector size will be derived with 32, which is the bit length of General Purpose Register in
        CUDA.

    Returns
    -------
    """
    from hidet.ir.tools import infer_type

    src_dtype = infer_type(src)
    bits_per_vector = 32 if elements_per_vector is None else src_dtype.nbits * elements_per_vector
    func_name = resolve_vectorized_cvt_func_name(src, dst, bits_per_vector)
    return call_primitive_func(func_name, args=[src, dst])


def cvtv_func(
    src: Union[Expr, DataType], dst: Union[Expr, DataType], elements_per_vector: Optional[int] = None
) -> Function:
    """
    Get the function that conver the src vector to the given data type

    Parameters
    ----------
    src: Union[Expr, DataType]
        The source expression or data type of the source expression to be converted.

    dst: Union[Expr, DataType]
        The destination expression or data type of the destination expression.

    elements_per_vector:
        The number of elements in a vector.
        By default, the vector size will be derived with 32, which is the bit length of General Purpose Register in
        CUDA.

    Returns
    -------
    ret: Function
        A function that can perform the conversion
    """
    from hidet.ir.tools import infer_type

    src_dtype = infer_type(src) if isinstance(src, Expr) else src
    bits_per_vector = 32 if elements_per_vector is None else src_dtype.nbits * elements_per_vector
    func_name = resolve_vectorized_cvt_func_name(src, dst, bits_per_vector)
    try:
        entry = lookup_primitive_function(func_name)
        return entry.var
    except ValueError:
        return None


def cast_u4x8_to_f16x8_interleaved_func() -> Function:
    func_name = 'cast_u4x8_to_f16x8_interleaved'
    try:
        entry = lookup_primitive_function(func_name)
        return entry.var
    except ValueError:
        return None
