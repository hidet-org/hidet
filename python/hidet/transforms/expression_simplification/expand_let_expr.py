from hidet.ir.expr import Let, var
from hidet.ir.stmt import *
from hidet.ir.functors import StmtExprRewriter
from hidet.transforms import Pass, FunctionBodyPass


def wrapper(stmt_visitor):
    def wrapped_visitor(self, stmt):
        self.stmt_stack.append([])
        self.memo.clear()  # do not cache exprs between different statements, so the let expr will always generate let stmt.
        updated_stmt = stmt_visitor(self, stmt)
        let_stmts = self.stmt_stack.pop()
        return LetStmt.concat_let_chain(let_stmts, updated_stmt)

    return wrapped_visitor


class LetExprExpander(StmtExprRewriter):

    def __init__(self):
        super().__init__()
        self.stmt_stack = []

    def expand(self, stmt):
        assert isinstance(stmt, Stmt)
        return self.visit(stmt)

    def visit_Let(self, e: Let):
        var = self(e.var)
        value = self(e.value)
        self.stmt_stack[-1].append(LetStmt(var, value))
        return self(e.body)

    @wrapper
    def visit_EvaluateStmt(self, stmt: EvaluateStmt):
        return StmtExprRewriter.visit_EvaluateStmt(self, stmt)

    @wrapper
    def visit_BufferStoreStmt(self, stmt: BufferStoreStmt):
        return StmtExprRewriter.visit_BufferStoreStmt(self, stmt)

    @wrapper
    def visit_AssignStmt(self, stmt: AssignStmt):
        return StmtExprRewriter.visit_AssignStmt(self, stmt)

    @wrapper
    def visit_LetStmt(self, stmt: LetStmt):
        return StmtExprRewriter.visit_LetStmt(self, stmt)

    @wrapper
    def visit_ForStmt(self, stmt: ForStmt):
        return StmtExprRewriter.visit_ForStmt(self, stmt)

    @wrapper
    def visit_IfStmt(self, stmt: IfStmt):
        return StmtExprRewriter.visit_IfStmt(self, stmt)

    @wrapper
    def visit_AssertStmt(self, stmt: AssertStmt):
        return StmtExprRewriter.visit_AssertStmt(self, stmt)

    @wrapper
    def visit_AsmStmt(self, stmt: AsmStmt):
        return StmtExprRewriter.visit_AsmStmt(self, stmt)

    @wrapper
    def visit_BlackBoxStmt(self, stmt: BlackBoxStmt):
        return StmtExprRewriter.visit_BlackBoxStmt(self, stmt)


class ExpandLetExprPass(FunctionBodyPass):
    def process_body(self, stmt: Stmt) -> Stmt:
        expander = LetExprExpander()
        stmt = expander.expand(stmt)
        return stmt


def expand_let_expr_pass() -> Pass:
    return ExpandLetExprPass()

