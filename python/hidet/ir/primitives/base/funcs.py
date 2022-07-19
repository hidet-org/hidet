from typing import List, Optional, Union
import builtins
import math
from hidet.ir.type import ScalarType
from hidet.ir.expr import Expr, Call, Var, cast
from hidet.ir.stmt import BlackBoxStmt, AsmStmt, ReturnStmt
from hidet.ir.builders import FunctionBuilder, StmtBuilder
from ..func import FuncType, register_primitive_function, is_primitive_function, lookup_primitive_function, registered_primitive_functions
from ..func import primitive_func_pool as pool
from hidet.utils import initialize

ExprLike = Union[Expr, int, float]


def call_base(name: str, args: List[ExprLike]) -> Call:
    entry = pool.lookup_by_name('base_{}'.format(name))
    if entry.func_type.type_infer_func is None:
        param_types = entry.func_type.param_types
        if len(param_types) != len(args):
            raise ValueError('Function {} expect {} arguments, got {}.'.format(name, len(param_types), len(args)))
    return Call(entry.var, args)


def max(a: ExprLike, b: ExprLike) -> Expr:
    return call_base('max', [a, b])


def min(a: ExprLike, b: ExprLike) -> Expr:
    return call_base('min', [a, b])


def exp(a: ExprLike) -> Expr:
    return call_base('exp', [a])


def pow(a: ExprLike, b: ExprLike) -> Expr:
    return call_base('pow', [a, b])


def sqrt(a: ExprLike) -> Expr:
    return call_base('sqrt', [a])


def rsqrt(a: ExprLike) -> Expr:
    return call_base('rsqrt', [a])


def erf(a: ExprLike) -> Expr:
    return call_base('erf', [a])


def sin(a: ExprLike) -> Expr:
    return call_base('sin', [a])


def cos(a: ExprLike) -> Expr:
    return call_base('cos', [a])


def tanh(a: ExprLike) -> Expr:
    return call_base('tanh', [a])


def round(a: ExprLike) -> Expr:
    return call_base('round', [a])


def floor(a: ExprLike) -> Expr:
    return call_base('floor', [a])


def ceil(a: ExprLike) -> Expr:
    return call_base('ceil', [a])


def printf(format_string, *args):
    """
    usage:
    printf(r"%d %d\n", expr_1, expr_2)
    """
    arg_string = ', '.join(['{}'] * len(args))
    template_string = f'printf("{format_string}", {arg_string});'
    return BlackBoxStmt(template_string, *args)


def type_infer_func(arg_types: List[ScalarType]) -> ScalarType:
    return builtins.max(arg_types)


@initialize()
def register_primitive_functions_generic():
    unary_names = [
        'neg', 'sin', 'cos', 'tanh', 'exp', 'round', 'floor', 'ceil', 'rsqrt', 'sqrt', 'erf'
    ]
    binary_names = [
        'min', 'max', 'pow'
    ]
    ternary_names = [
        'fma'
    ]
    for name in unary_names + binary_names + ternary_names:
        register_primitive_function(
            name='{}_{}'.format('base', name),
            codegen_name=None,
            func_or_type=FuncType(type_infer_func=type_infer_func),
            generic=True
        )


@initialize()
def register_primitive_functions_float32():
    unary_names = [
        'sinf', 'cosf', 'tanhf', 'expf', 'roundf', 'floorf', 'ceilf', 'rsqrtf', 'sqrtf', 'erff'
    ]
    binary_names = [
        'fminf', 'fmaxf', 'powf'
    ]
    ternary_names = [
        'fmaf'
    ]
    base2float32 = {
        'sin': 'sinf',
        'cos': 'cosf',
        'tanh': 'tanhf',
        'exp': 'expf',
        'round': 'roundf',
        'floor': 'floorf',
        'ceil': 'ceilf',
        'rsqrt': 'rsqrtf',
        'erf': 'erff',
        'sqrt': 'sqrtf',
        'min': 'fminf',
        'max': 'fmaxf',
        'pow': 'powf',
        'fma': 'fmaf'
    }
    for names, param_types in zip([unary_names, binary_names, ternary_names], [['float32'], ['float32'] * 2, ['float32'] * 3]):
        for name in names:
            register_primitive_function(
                name='{}_{}'.format('cuda_fp32', name),
                codegen_name=name,
                func_or_type=FuncType(param_types=param_types, ret_type='float32'),
                generic=False
            )
    for a, b in base2float32.items():
        base_name = '{}_{}'.format('base', a)
        fp32_name = '{}_{}'.format('cuda_fp32', b)
        pool.lookup_by_name(base_name).dispatch_dtype(dtype='float32', dispatched_func_name=fp32_name)


@initialize()
def register_primitive_functions_int32():
    binary_names = [
        'max', 'min'
    ]
    base2int32 = {
        'max': 'max',
        'min': 'min'
    }
    for name in binary_names:
        register_primitive_function(
            name='{}_{}'.format('cuda_int32', name),
            codegen_name=name,
            func_or_type=FuncType(param_types=['int32', 'int32'], ret_type='int32'),
            generic=False
        )
    for a, b in base2int32.items():
        base_name = '{}_{}'.format('base', a)
        int32_name = '{}_{}'.format('int32', b)
        pool.lookup_by_name(base_name).dispatch_dtype(dtype='int32', dispatched_func_name=int32_name)

