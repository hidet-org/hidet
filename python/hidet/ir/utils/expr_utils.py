from typing import Union
from hidet.ir.expr import Expr, Constant


def as_expr(e: Union[int, float, bool, Expr]):
    if isinstance(e, Expr):
        return e
    elif isinstance(e, int):
        return Constant(value=e, const_type='int32')
    elif isinstance(e, float):
        return Constant(value=e, const_type='float32')
    elif isinstance(e, bool):
        return Constant(value=e, const_type='bool')
    else:
        raise ValueError('Cannot convert {} to hidet.ir.Expr.'.format(e))
