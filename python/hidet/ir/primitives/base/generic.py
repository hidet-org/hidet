from typing import List, Union
from hidet.ir.type import DataType
from hidet.ir.expr import Expr, Call
from hidet.ir.stmt import BlackBoxStmt
from hidet.utils import initialize
from ..func import FuncType, register_primitive_function
from ..func import primitive_func_pool as pool

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


def log(a: ExprLike) -> Expr:
    return call_base('log', [a])


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
    if len(args) > 0:
        arg_string = ', '.join(['{}'] * len(args))
        template_string = f'printf("{format_string}", {arg_string});'
    else:
        template_string = f'printf("{format_string}");'
    if '\n' in format_string:
        raise ValueError('Please use printf(r"...\\n") instead of printf("...\\n").')
    return BlackBoxStmt(template_string, *args)


def type_infer_func(arg_types: List[DataType]) -> DataType:
    from hidet.ir.utils.type_utils import numeric_promotation

    dtype = arg_types[0]
    for arg_type in arg_types[1:]:
        dtype = numeric_promotation(dtype, arg_type)
    return dtype


@initialize()
def register_primitive_functions_generic():
    unary_names = ['neg', 'sin', 'cos', 'tanh', 'exp', 'round', 'floor', 'ceil', 'rsqrt', 'sqrt', 'erf', 'log']
    binary_names = ['min', 'max', 'pow']
    ternary_names = ['fma']
    for name in unary_names + binary_names + ternary_names:
        register_primitive_function(
            name='{}_{}'.format('base', name),
            codegen_name=None,
            func_or_type=FuncType(type_infer_func=type_infer_func),
            generic=True,
        )
