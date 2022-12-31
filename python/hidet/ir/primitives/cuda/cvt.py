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
from hidet.utils import initialize
from hidet.ir.type import DataType
from hidet.ir.expr import Expr
from hidet.ir.stmt import asm
from hidet.ir.primitives.func import register_primitive_function, call_primitive_func
from hidet.lang import script


def resolve_cvt_func_name(src: Union[Expr, DataType], dtype: DataType) -> str:
    from hidet.ir.functors import infer_type

    if isinstance(src, DataType):
        src_dtype = src
    else:
        src_dtype = infer_type(src)
    if not isinstance(src_dtype, DataType):
        raise TypeError('src must be a scalar data type, got {}'.format(src_dtype))
    return 'cuda_cvt_{}_to_{}'.format(src_dtype.short_name, dtype.short_name)


@initialize()
def register_cvta_instructions():
    from hidet.lang import attr
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
                attr.func_name = func_name
                attr.func_kind = 'cuda_device'
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
