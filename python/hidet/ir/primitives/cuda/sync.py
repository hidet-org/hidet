from hidet.ir.type import ScalarType
from typing import List, Optional, Union, Tuple

from hidet.ir.builders import FunctionBuilder, StmtBuilder
from hidet.ir.dialects.lowlevel import PointerType, ReferenceType
from hidet.ir.dialects.lowlevel import VoidType
from hidet.ir.expr import Expr, Call, cast
from hidet.ir.expr import Var
from hidet.ir.stmt import AsmStmt, BlackBoxStmt, ReturnStmt
from hidet.ir.type import ScalarType, FuncType
from hidet.ir.func import Function
from hidet.ir.primitives.func import register_primitive_function, primitive_func_pool
from hidet.utils import initialize
from hidet.ir.primitives.func import call_primitive_func


@initialize()
def register_primitive_functions():
    functions = [
        ('cuda_syncthreads', '__syncthreads', FuncType([], VoidType())),
        ('cuda_syncthreads_count', '__syncthreads_count', FuncType(['int32'], 'int32')),
        ('cuda_syncthreads_and', '__syncthreads_and', FuncType(['int32'], 'int32')),
        ('cuda_syncthreads_or', '__syncthreads_or', FuncType(['int32'], 'int32')),
        ('cuda_syncwarp', '__syncwarp', FuncType([], VoidType()))
    ]
    for name, codegen_name, func_type in functions:
        register_primitive_function(name=name, func_or_type=func_type, codegen_name=codegen_name)


def syncthreads() -> Call:
    return call_primitive_func('cuda_syncthreads', [])


def syncthreads_count(value: Expr) -> Call:
    return call_primitive_func('cuda_syncthreads_count', [value])


def syncthreads_and(cond: Union[Expr, int, bool]) -> Call:
    return call_primitive_func('cuda_syncthreads_and', [cond])


def syncthreads_or(cond: Expr) -> Call:
    return call_primitive_func('cuda_syncthreads_or', [cond])


def syncwarp() -> Call:
    return call_primitive_func('cuda_syncwarp', [])

