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
from hidet.ir.dtypes import i32
from hidet.ir.expr import Expr, cast
from hidet.ir.primitives.func import register_primitive_function, call_primitive_func
from hidet.utils import initialize


@initialize()
def register_fastdiv_functions():
    from hidet.lang import script, attrs, asm

    dtype = i32
    div_func_name = 'fastintdiv'

    @script
    def div_op(dividend: dtype, divisor: dtype, m: dtype, s: dtype, ads: dtype) -> dtype:
        attrs.func_kind = 'cuda_internal'
        attrs.func_name = div_func_name
        q = 0
        asm('mul.hi.s32 %0, %1, %2;', outputs=[q], inputs=[m, dividend])
        q = q + dividend * ads
        if s >= 0:
            q = q >> s
            q = q + (cast(q, 'uint32') >> 31)
        return q

    register_primitive_function(name=div_func_name, func_or_type=div_op)

    mod_func_name = 'fastintmod'

    @script
    def mod_op(dividend: dtype, divisor: dtype, m: dtype, s: dtype, ads: dtype) -> dtype:
        attrs.func_kind = 'cuda_internal'
        attrs.func_name = mod_func_name
        q = 0
        asm('mul.hi.s32 %0, %1, %2;', outputs=[q], inputs=[m, dividend])
        q = q + dividend * ads
        if s >= 0:
            q = q >> s
            q = q + (cast(q, 'uint32') >> 31)
        remainder = dividend - q * divisor
        return remainder

    register_primitive_function(name=mod_func_name, func_or_type=mod_op)


# fast int div and fast int mod's implementation are borrowed from:
# https://github.com/milakov/int_fastdiv
def fast_intdiv(dividend: Expr, divisor: Expr, m: int, s: int, ads: int):
    return call_primitive_func('fastintdiv', [dividend, divisor, m, s, ads])


def fast_intmod(dividend: Expr, divisor: Expr, m: int, s: int, ads: int):
    return call_primitive_func('fastintdiv', [dividend, divisor, m, s, ads])
