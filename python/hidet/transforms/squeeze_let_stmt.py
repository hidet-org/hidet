from hidet.ir.stmt import Stmt, LetStmt
from hidet.ir.functors import StmtRewriter, same_list
from .base import FunctionBodyPass


class SqueezeLetStmtRewriter(StmtRewriter):
    def visit_SeqLetStmt(self, stmt: LetStmt):
        bind_vars = []
        bind_values = []
        cur = stmt
        while isinstance(cur, LetStmt):
            bind_vars.extend(cur.bind_vars)
            bind_vars.extend(cur.bind_values)
            cur = cur.body
        if same_list(bind_vars, stmt.bind_vars) and same_list(bind_values, stmt.bind_values) and cur is stmt.body:
            return stmt
        else:
            return LetStmt(bind_vars, bind_values, stmt)


class SqueezeLetStmtPass(FunctionBodyPass):
    def process_body(self, stmt: Stmt) -> Stmt:
        return SqueezeLetStmtRewriter()(stmt)


def squeeze_let_stmt_pass():
    return SqueezeLetStmtPass()
