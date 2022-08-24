from typing import Union

from hidet.ir.expr import Expr, convert
from hidet.ir.func import Function
from hidet.ir.type import FuncType, ScalarType
from hidet.ir.primitives.func import register_primitive_function, call_primitive_func
from hidet.utils import initialize


@initialize()
def register_functions():
    i32 = ScalarType('int32')
    register_primitive_function('cuda_atomic_add', func_or_type=FuncType([~i32, i32], i32), codegen_name='atomicAdd')
    register_primitive_function('cuda_atomic_sub', func_or_type=FuncType([~i32, i32], i32), codegen_name='atomicSub')
    register_primitive_function('cuda_atomic_exchange', func_or_type=FuncType([~i32, i32], i32), codegen_name='atomicExch')
    register_primitive_function('cuda_atomic_cas', func_or_type=FuncType([~i32, i32, i32], i32), codegen_name='atomicCAS')


def atomic_add(addr: Expr, value: Expr):
    return call_primitive_func('cuda_atomic_add', [addr, value])


def atomic_sub(addr: Expr, value: Expr):
    return call_primitive_func('cuda_atomic_sub', [addr, value])


def atomic_exchange(addr: Expr, value: Expr):
    return call_primitive_func('cuda_atomic_exchange', [addr, value])


def atomic_cas(addr: Expr, compare: Union[Expr, int], value: Union[Expr, int]):
    return call_primitive_func('cuda_atomic_cas', [addr, compare, value])

