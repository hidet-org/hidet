from hidet.ir.expr import IfThenElse
from hidet.ir.stmt import Stmt, IfStmt
from hidet.ir.func import Function

from .base import FunctionPass
from .common import FuncStmtExprRewriterWithScope


class PrecomputeConditionRewriter(FuncStmtExprRewriterWithScope):
    def __init__(self):
        super().__init__(use_memo=False)

    def visit_IfStmt(self, stmt: IfStmt):
        scope = self.scope_stack.current()
        cond = scope.define_predicate(stmt.cond)
        then_body = self.visit(stmt.then_body)
        else_body = self.visit(stmt.else_body) if stmt.else_body else None
        return IfStmt(cond, then_body, else_body)

    def visit_IfThenElse(self, e: IfThenElse):
        scope = self.scope_stack.current()
        cond = scope.define_predicate(e.cond)
        return IfThenElse(cond, e.then_expr, e.else_expr)


class PrecomputeConditionPass(FunctionPass):
    def process_func(self, func: Function) -> Function:
        rewriter = PrecomputeConditionRewriter()
        return rewriter(func)


def precompute_condition_pass():
    return PrecomputeConditionPass()
