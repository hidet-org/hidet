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

i32 = data_type('int32')
u32 = data_type('uint32')
i64 = data_type('int64')
f32 = data_type('float32')
f16 = data_type('float16')

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
