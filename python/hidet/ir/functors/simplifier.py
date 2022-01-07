from typing import Union
import operator
from hidet.ir.expr import Expr, BinaryOp, Add, Sub, Multiply, Div, Mod, FloorDiv, LessThan, Equal, Constant
from hidet.ir.expr import is_one, is_zero, convert
from hidet.ir.stmt import Stmt
from hidet.ir.functors import StmtExprRewriter


class Simplifier(StmtExprRewriter):
    def visit_Binary(self, e: BinaryOp):
        a = self(e.a)
        b = self(e.b)
        if isinstance(e, Add):
            if is_zero(a):
                return b
            if is_zero(b):
                return a
        elif isinstance(e, Sub):
            if is_zero(b):
                return a
        elif isinstance(e, Multiply):
            if is_one(a):
                return b
            if is_one(b):
                return a
        elif isinstance(e, Div):
            if is_one(b):
                return a
        elif isinstance(e, Mod):
            pass
        elif isinstance(e, FloorDiv):
            if is_one(b):
                return a
        elif isinstance(e, LessThan):
            pass
        elif isinstance(e, Equal):
            pass
        else:
            raise ValueError()

        if isinstance(a, Constant) and isinstance(b, Constant):
            op_dict = {
                Add: operator.add,
                Sub: operator.sub,
                Multiply: operator.mul,
                Div: operator.truediv,
                Mod: operator.mod,
                FloorDiv: operator.floordiv,
                LessThan: operator.lt,
                Equal: operator.eq
            }
            return convert(op_dict[e.__class__](a.value, b.value))
        if a is e.a and b is e.b:
            return e
        return e.__class__(a, b)

    def visit_Add(self, e: Add):
        return self.visit_Binary(e)

    def visit_Sub(self, e: Sub):
        return self.visit_Binary(e)

    def visit_Multiply(self, e: Multiply):
        return self.visit_Binary(e)

    def visit_Div(self, e: Div):
        return self.visit_Binary(e)

    def visit_Mod(self, e: Mod):
        return self.visit_Binary(e)

    def visit_FloorDiv(self, e: FloorDiv):
        return self.visit_Binary(e)

    def visit_LessThan(self, e: LessThan):
        return self.visit_Binary(e)

    def visit_Equal(self, e: Equal):
        return self.visit_Binary(e)


def simplify(node: Union[Stmt, Expr]):
    simplifier = Simplifier()
    return simplifier(node)
