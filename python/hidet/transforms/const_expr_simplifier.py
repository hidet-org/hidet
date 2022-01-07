from hidet.ir.functors import simplify
from hidet.ir.expr import Expr
from hidet.ir.func import Function, IRModule
from hidet.transforms.base import Pass


class ConstExprSimplifier(Pass):
    def __init__(self):
        super().__init__('const_expr_simplifier')

    def process_func(self, func: Function):
        body = simplify(func.body)
        if body is not func.body:
            return Function(func.name, func.params, body, func.ret_type, func.local_vars, func.attrs)
        else:
            return func


def const_expr_simplifier():
    return ConstExprSimplifier()
