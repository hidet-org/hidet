from hidet.ir.expr import IfThenElse
from hidet.ir.stmt import Stmt, IfStmt, ForStmt
from hidet.ir.func import Function

from .base import FunctionPass
from .common import FuncStmtExprRewriterWithScope


class PrecomputeConditionRewriter(FuncStmtExprRewriterWithScope):
    def __init__(self):
        super().__init__(use_memo=False)

    @staticmethod
    def scope_in_loop(scope) -> bool:
        while scope is not None:
            if isinstance(scope.scope_stmt, ForStmt):
                return True
            scope = scope.parent
        return False

    def should_precompute(self, cond) -> bool:
        """
        we only precompute when the following two conditions holds:
            1. current scope is in a loop.
            2. the used variables are defined not in any loop
        """
        return self.scope_in_loop(self.scope_stack.current()) and not self.scope_in_loop(self.scope_to_define(cond))

    def visit_IfStmt(self, stmt: IfStmt):
        if self.should_precompute(stmt.cond):
            # we can precompute the predicate
            scope = self.scope_to_define(stmt.cond)
            cond = scope.define_predicate(stmt.cond)
            then_body = self.visit(stmt.then_body)
            else_body = self.visit(stmt.else_body) if stmt.else_body else None
            return IfStmt(cond, then_body, else_body)
        else:
            return FuncStmtExprRewriterWithScope.visit_IfStmt(self, stmt)

    def visit_IfThenElse(self, e: IfThenElse):
        if self.should_precompute(e.cond):
            scope = self.scope_to_define(e.cond)
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
