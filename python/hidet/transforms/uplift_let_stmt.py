from hidet.ir.func import Function
from hidet.ir.stmt import LetStmt
from .base import FunctionPass
from .common import FuncStmtExprRewriterWithScope


class UpliftLetStmtRewriter(FuncStmtExprRewriterWithScope):
    def visit_LetStmt(self, stmt: LetStmt):
        with self.new_scope(stmt) as scope:
            for var, value in zip(stmt.bind_vars, stmt.bind_values):
                value = self.visit(value)
                scope_to_define = self.scope_to_define(value)
                scope_to_define.define(var, value)
            return scope.wrap(self.visit(stmt.body))


class UpliftLetStmtPass(FunctionPass):
    def process_func(self, func: Function) -> Function:
        rewriter = UpliftLetStmtRewriter()
        return rewriter.visit(func)


def uplift_let_stmt_pass():
    return UpliftLetStmtPass()
