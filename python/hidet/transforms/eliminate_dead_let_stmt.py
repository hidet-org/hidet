from collections import defaultdict
from hidet.ir.expr import Var
from hidet.ir.stmt import LetStmt
from hidet.ir.functors import StmtExprRewriter, rewrite
from hidet.ir.func import Function
from hidet.transforms import Pass, FunctionPass


class DeadLetStmtEliminator(StmtExprRewriter):
    def __init__(self):
        super().__init__()
        self.ref_count = defaultdict(int)

    def visit(self, obj):
        if isinstance(obj, Var):
            self.ref_count[obj] += 1
        return StmtExprRewriter.visit(self, obj)

    def eliminate(self, stmt):
        self.memo.clear()
        self.ref_count.clear()
        return self.visit(stmt)

    def visit_LetStmt(self, stmt: LetStmt):
        value = self.visit(stmt.value)
        body = self.visit(stmt.body)
        var = stmt.var
        if self.ref_count[var] == 0:
            return body
        elif self.ref_count[var] == 1:
            # replace the var that has used only once.
            # this is a naive O(N^2) method to replace, change to O(N) when needed
            return rewrite(body, {var: value})
        else:
            if body is stmt.body and value is stmt.value:
                return stmt
            else:
                return LetStmt(stmt.var, value, body)

    def visit_Var(self, e: Var):
        return e


class EliminateDeadLetStmtPass(FunctionPass):
    def process_func(self, func: Function) -> Function:
        eliminator = DeadLetStmtEliminator()
        body = func.body
        while True:
            orig_body = body
            body = eliminator.eliminate(body)
            if orig_body is body:
                break
        if body is func.body:
            return func
        else:
            return Function(func.name, func.params, body, func.ret_type, func.local_vars, func.attrs)


def eliminate_dead_let_stmt_pass() -> Pass:
    return EliminateDeadLetStmtPass()
