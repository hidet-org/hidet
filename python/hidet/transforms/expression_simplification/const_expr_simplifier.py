import operator

from hidet.ir import Stmt
from hidet.ir.expr import Add, convert, Sub, Multiply, FloorDiv, Mod, LessThan, LessEqual, Equal, BinaryOp
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


class ConstExprSimplifyPass(FunctionBodyPass):
    def process_body(self, stmt: Stmt) -> Stmt:
        return ConstExprSimplifier()(stmt)


def const_expr_simplify_pass():
    return ConstExprSimplifyPass()
