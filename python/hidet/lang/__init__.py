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
from typing import Union, Sequence, Optional, List
from hidet.ir.type import TypeNode, DataType, TensorType, PointerType, VoidType, ReferenceType, void_p, data_type
from hidet.ir.expr import Expr, Var, cast, view, Dereference
from hidet.ir.mapping import row_spatial, row_repeat, col_repeat, col_spatial, TaskMapping, auto_map
from hidet.ir.layout import DataLayout
from hidet.ir.primitives import printf
from hidet.lang.script import script, script_module
from hidet.ir.stmt import asm, DeclareScope
from hidet.ir.func import Function
from hidet.lang.type_utils import static, with_scope
from hidet.ir.dtypes import int8, int16, int32, int64, uint8, uint16, uint32, uint64, float16, float32, float64, boolean
from hidet.ir.dtypes import i8, i16, i32, i64, u8, u16, u32, u64, f16, f32, f64
from hidet.ir.dtypes import bfloat16, tfloat32, bf16, tf32


ref_u32 = ReferenceType(u32)

void_pointer = PointerType(VoidType())
void = VoidType()

spatial = row_spatial
repeat = row_repeat


ConstExpr = Union[Expr, int]


def tensor(
    scope: Union[DeclareScope, str],
    dtype: Union[DataType, str],
    shape: Optional[Sequence[ConstExpr]] = None,
    layout: Optional[DataLayout] = None,
):
    from hidet.ir.type import tensor_type

    return with_scope(scope, tensor_type(dtype, shape, layout))


def tensor_pointer(
    dtype: Union[DataType, str], shape: Optional[Sequence[ConstExpr]] = None, layout: Optional[DataLayout] = None
):
    # pylint: disable=import-outside-toplevel
    from hidet.ir.type import tensor_type

    return ~tensor_type(dtype, shape, layout)


def as_tensor_pointer(
    expr: Expr,
    dtype: Union[DataType, str],
    shape: Optional[Sequence[ConstExpr]] = None,
    layout: Optional[DataLayout] = None,
) -> Expr:
    return cast(expr, tensor_pointer(dtype, shape, layout))


def grid(*dim_extents):
    raise ValueError('Please call this function within the @hidet.script decorated function.')


def deref(addr: Expr):
    return Dereference(addr)


def var_of_function(func: Function) -> Var:
    # pylint: disable=import-outside-toplevel
    from hidet.lang.script import ScriptModuleContext

    if not isinstance(func, Function):
        raise ValueError('Expect a hidet.ir.Function, got {}.'.format(type(func).__name__))
    ctx = ScriptModuleContext.current_context()
    func_var: Optional[Var] = ctx.lookup(func.name)
    if func_var is None:
        raise ValueError('Function has not been defined in current script module.')
    return func_var
