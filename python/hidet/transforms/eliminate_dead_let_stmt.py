from hidet.ir.stmt import LetStmt
from hidet.ir.functors import StmtExprRewriter
from hidet.ir.func import Function
from hidet.transforms import Pass, FunctionPass


class DeadLetStmtEliminator(StmtExprRewriter):
    def eliminate(self, stmt):
        return self.visit(stmt)

    def visit_LetStmt(self, stmt: LetStmt):
        value = self.visit(stmt.value)
        body = self.visit(stmt.body)
        if stmt.var not in self.memo:
            # unused let var
            return body
        else:
            if body is stmt.body and value is stmt.value:
                return stmt
            else:
                return LetStmt(stmt.var, value, body)


def eliminate_dead_let_stmt_pass() -> Pass:
    def process_func(func: Function) -> Function:
        eliminator = DeadLetStmtEliminator()
        body = eliminator.eliminate(func.body)
        if body is func.body:
            return func
        else:
            return Function(func.name, func.params, body, func.ret_type, func.local_vars, func.attrs)

    return FunctionPass('dead_let_stmt_eliminator', process_func)
