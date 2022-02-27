from hidet.ir.stmt import Stmt, LetStmt, SeqLetStmt
from hidet.ir.functors import StmtRewriter, same_list
from .base import FunctionBodyPass


class SqueezeLetStmtRewriter(StmtRewriter):
    def visit_LetStmt(self, stmt: LetStmt):
        bind_vars = []
        bind_values = []
        while isinstance(stmt, (LetStmt, SeqLetStmt)):
            if isinstance(stmt, LetStmt):
                bind_vars.append(stmt.var)
                bind_values.append(stmt.value)
            else:
                bind_vars.extend(stmt.bind_vars)
                bind_vars.extend(stmt.bind_values)
            stmt = stmt.body
        return SeqLetStmt(bind_vars, bind_values, stmt)

    def visit_SeqLetStmt(self, stmt: SeqLetStmt):
        bind_vars = []
        bind_values = []
        cur = stmt
        while isinstance(cur, (LetStmt, SeqLetStmt)):
            if isinstance(cur, LetStmt):
                bind_vars.append(cur.var)
                bind_values.append(cur.value)
            else:
                bind_vars.extend(cur.bind_vars)
                bind_vars.extend(cur.bind_values)
            cur = cur.body
        if same_list(bind_vars, stmt.bind_vars) and same_list(bind_values, stmt.bind_values) and cur is stmt.body:
            return stmt
        else:
            return SeqLetStmt(bind_vars, bind_values, stmt)


class SqueezeLetStmtPass(FunctionBodyPass):
    def process_body(self, stmt: Stmt) -> Stmt:
        return SqueezeLetStmtRewriter()(stmt)


def squeeze_let_stmt_pass():
    return SqueezeLetStmtPass()
