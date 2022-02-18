import operator

from hidet.ir import Stmt
from hidet.ir.expr import Add, convert, Sub, Multiply, FloorDiv, Mod, LessThan, LessEqual, Equal, BinaryOp, And
from hidet.ir.functors import StmtExprRewriter
from hidet.transforms.base import FunctionBodyPass


class ConstExprSimplifier(StmtExprRewriter):
    op_dict = {
        Add: operator.add,
        Sub: operator.sub,
        Multiply: operator.mul,
        FloorDiv: operator.floordiv,
        Mod: operator.mod,
        LessThan: operator.lt,
        LessEqual: operator.le,
        Equal: operator.eq,
    }

    def visit_Binary(self, e: BinaryOp):
        e = StmtExprRewriter.visit_Binary(self, e)
        if e.a.is_const() and e.b.is_const() and e.__class__ in self.op_dict:
            op = self.op_dict[e.__class__]
            return convert(op(e.a.const().value, e.b.const().value))
        return e

    def visit_And(self, e: And):
        e = StmtExprRewriter.visit_Binary(self, e)
        a_val = e.a.const().value if e.a.is_const() else None
        b_val = e.b.const().value if e.b.is_const() else None
        if a_val and b_val:
            return convert(True)
        elif a_val is False or b_val is False:
            return convert(False)
        elif a_val:
            return e.b
        elif b_val:
            return e.a
        else:
            return e


class ConstExprSimplifyPass(FunctionBodyPass):
    def process_body(self, stmt: Stmt) -> Stmt:
        return ConstExprSimplifier()(stmt)


def const_expr_simplify_pass():
    return ConstExprSimplifyPass()
