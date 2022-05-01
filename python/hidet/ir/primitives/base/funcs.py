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
    entry = pool.lookup_by_name(target='base', name=name)
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
    # level = {
    #     'float64': 10,
    #     'float32': 9,
    #     'bfloat16': 8,
    #     'float16': 7,
    #
    #     'int64': 5,
    #     'uint64': 4.5,
    #     'int32': 4,
    #     'uint32': 3.5,
    #     'int16': 3,
    #     'uint16': 2.5,
    #     'int8': 2,
    #     'uint8': 1.5
    # }
    # return list(sorted(arg_types, key=lambda a: level[a.name]))[-1]
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
    for unary in unary_names:
        register_primitive_function('base', unary, FuncType(type_infer_func=type_infer_func), generic=True)
    for binary in binary_names:
        register_primitive_function('base', binary, FuncType(type_infer_func=type_infer_func), generic=True)
    for ternary in ternary_names:
        register_primitive_function('base', ternary, FuncType(type_infer_func=type_infer_func), generic=True)


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
    for unary in unary_names:
        register_primitive_function('base', unary, FuncType(param_types=['float32'], ret_type='float32'))
    for binary in binary_names:
        register_primitive_function('base', binary, FuncType(param_types=['float32', 'float32'], ret_type='float32'))
    for ternary in ternary_names:
        register_primitive_function('base', ternary, FuncType(param_types=['float32', 'float32', 'float32'], ret_type='float32'))
    for base_name, fp32_name in base2float32.items():
        pool.lookup_by_name('base', base_name).dispatch_dtype(dtype='float32', space='base', func_name=fp32_name)


