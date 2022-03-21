from hidet.ir.expr import IfThenElse
from hidet.ir.stmt import Stmt, IfStmt
from hidet.ir.func import Function

from .base import FunctionPass
from .common import FuncStmtExprRewriterWithScope


class PrecomputeConditionRewriter(FuncStmtExprRewriterWithScope):
    def __init__(self):
        super().__init__(use_memo=False)

    def should_precompute(self, cond) -> bool:
        return self.scope_stack.current().find_scope_for_expr(cond) is self.scope_stack.scopes[0]

    def visit_IfStmt(self, stmt: IfStmt):
        if self.should_precompute(stmt.cond):
            # we can precompute the predicate
            scope = self.scope_stack.current()
            cond = scope.define_predicate(stmt.cond)
            then_body = self.visit(stmt.then_body)
            else_body = self.visit(stmt.else_body) if stmt.else_body else None
            return IfStmt(cond, then_body, else_body)
        else:
            return FuncStmtExprRewriterWithScope.visit_IfStmt(self, stmt)

    def visit_IfThenElse(self, e: IfThenElse):
        if self.should_precompute(e.cond):
            scope = self.scope_stack.current()
            cond = scope.define_predicate(e.cond)
            return IfThenElse(cond, e.then_expr, e.else_expr)
        else:
            return FuncStmtExprRewriterWithScope.visit_IfThenElse(self, e)


class PrecomputeConditionPass(FunctionPass):
    def process_func(self, func: Function) -> Function:
        rewriter = PrecomputeConditionRewriter()
        return rewriter(func)


def precompute_condition_pass():
    return PrecomputeConditionPass()
