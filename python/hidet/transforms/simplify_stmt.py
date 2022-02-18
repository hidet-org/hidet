from hidet.ir import Stmt
from hidet.ir.expr import is_one, is_zero, is_true, is_false, convert
from hidet.ir.stmt import IfStmt, ForStmt, SeqStmt
from hidet.ir.functors import StmtExprRewriter
from hidet.transforms.base import FunctionBodyPass


class StatementSimplifier(StmtExprRewriter):
    def visit_IfStmt(self, stmt: IfStmt):
        if is_true(stmt.cond):
            then_body = self(stmt.then_body)
            return then_body
        elif is_false(stmt.cond):
            if stmt.else_body:
                return self(stmt.else_body)
            else:
                return SeqStmt([])
        else:
            return StmtExprRewriter.visit_IfStmt(self, stmt)

    def visit_ForStmt(self, stmt: ForStmt):
        if is_zero(stmt.extent):
            return SeqStmt([])
        elif is_one(stmt.extent):
            self.memo[stmt.loop_var] = convert(0)
            return self(stmt.body)
        else:
            return StmtExprRewriter.visit_ForStmt(self, stmt)


class SimplifyStmtPass(FunctionBodyPass):
    def process_body(self, stmt: Stmt) -> Stmt:
        return StatementSimplifier()(stmt)


def simplify_stmt_pass():
    return SimplifyStmtPass()
