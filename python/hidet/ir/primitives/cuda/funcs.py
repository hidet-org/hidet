from hidet.ir.type import ScalarType
from typing import List, Optional, Union, Tuple

from hidet.ir.builders import FunctionBuilder, StmtBuilder
from hidet.ir.expr import Expr, Call, cast
from hidet.ir.expr import Var
from hidet.ir.stmt import AsmStmt, BlackBoxStmt, ReturnStmt
from hidet.ir.type import ScalarType, FuncType, PointerType, ReferenceType, VoidType
from hidet.ir.func import Function
from hidet.ir.primitives.func import register_primitive_function, primitive_func_pool
from hidet.utils import initialize


def register_unary_dialect_primitive_function(func_name, generic_func, target_dtype: str, dialect_dtype: str):
    with FunctionBuilder(func_name, kind='cuda_device', ret_type=ScalarType(target_dtype)) as fb:
        # params
        x = Var('x', type=ScalarType(target_dtype))
        fb.extend_params([x])
        # body
        sb = StmtBuilder()
        sb += ReturnStmt(cast(generic_func(cast(x, dialect_dtype)), target_dtype))
        fb.set_body(sb.finish())
    register_primitive_function(name=func_name, func_or_type=fb.get())


def register_binary_dialect_primitive_function(func_name, generic_func, target_dtype: str, dialect_dtype: str):
    with FunctionBuilder(func_name, kind='cuda_device', ret_type=ScalarType(target_dtype)) as fb:
        # params
        x = Var('x', type=ScalarType(target_dtype))
        y = Var('y', type=ScalarType(target_dtype))
        fb.extend_params([x, y])
        # body
        sb = StmtBuilder()
        sb += ReturnStmt(cast(generic_func(cast(x, dialect_dtype), cast(y, dialect_dtype)), target_dtype))
        fb.set_body(sb.finish())
    register_primitive_function(name=func_name, func_or_type=fb.get())


def call_cuda(func_name, args: List[Expr]) -> Call:
    # todo: replace all usage of this function to call_primitive_func
    entry = primitive_func_pool.lookup_by_name('cuda_{}'.format(func_name))
    return Call(entry.var, args)

