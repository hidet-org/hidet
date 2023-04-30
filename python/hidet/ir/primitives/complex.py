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
from hidet.ir.type import FuncType, DataType
from hidet.ir.dtypes import complex64, complex128
from hidet.ir.expr import Expr
from hidet.ir.primitives.func import register_primitive_function, call_primitive_func
from hidet.utils import initialize


def _real_imag_type_infer(arg_types) -> DataType:
    arg_type: DataType = arg_types[0]
    if arg_type.is_complex():
        from hidet.ir.dtypes.complex import ComplexType

        assert isinstance(arg_type, ComplexType)
        return arg_type.base_dtype
    else:
        raise TypeError('The argument of real/imag must be complex, got {}.'.format(arg_type))


def _make_complex_type_infer(arg_types) -> DataType:
    lhs_dtype, rhs_dtype = arg_types
    if lhs_dtype != rhs_dtype:
        raise TypeError(
            'The argument of make_complex must be the same type, got {} and {}.'.format(lhs_dtype, rhs_dtype)
        )
    else:
        if lhs_dtype.name == 'float32':
            return complex64
        elif lhs_dtype.name == 'float64':
            return complex128
        else:
            raise TypeError('The argument of make_complex must be float32 or float64, got {}.'.format(lhs_dtype))


@initialize()
def register_complex_functions():
    register_primitive_function(
        name='imag', codegen_name='imag', func_or_type=FuncType(type_infer_func=_real_imag_type_infer)
    )
    register_primitive_function(
        name='real', codegen_name='real', func_or_type=FuncType(type_infer_func=_real_imag_type_infer)
    )
    register_primitive_function(
        name='conj', codegen_name='conj', func_or_type=FuncType(type_infer_func=lambda arg_types: arg_types[0])
    )
    register_primitive_function(
        name='make_complex',
        codegen_name='make_complex',
        func_or_type=FuncType(type_infer_func=_make_complex_type_infer),
    )


def imag(a: Expr) -> Expr:
    return call_primitive_func('imag', [a])


def real(a: Expr) -> Expr:
    return call_primitive_func('real', [a])


def conj(a: Expr) -> Expr:
    return call_primitive_func('conj', [a])


def make_complex(a: Expr, b: Expr) -> Expr:
    return call_primitive_func('make_complex', [a, b])
