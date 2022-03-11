import contextlib
from hidet.ir.expr import *
from hidet.ir.stmt import *
from hidet.ir.functors import StmtExprRewriter, StmtRewriter, same_list, TypeInfer
from hidet.ir.builders import StmtBuilder
from hidet.transforms.base import FunctionBodyPass


class StmtContext:
    def __init__(self, rewriter: 'BuildLetStmtRewriter'):
        self.rewriter = rewriter

    def __enter__(self):
        self.rewriter.exit_stack_list.append(contextlib.ExitStack())
        self.rewriter.exit_stack = self.rewriter.exit_stack_list[-1]
        self.rewriter.exit_stack.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        exit_stack = self.rewriter.exit_stack_list.pop()
        exit_stack.__exit__(exc_type, exc_val, exc_tb)


class BuildLetStmtRewriter(StmtExprRewriter):
    def __init__(self):
        super().__init__(use_memo=False)
        self.exit_stack_list = []
        self.exit_stack: Optional[contextlib.ExitStack] = None
        self.sb: Optional[StmtBuilder] = None
        self.type_infer = TypeInfer()

    def build(self, stmt):
        self.sb = StmtBuilder()
        with StmtContext(self):
            self(stmt)
        return self.sb.finish()

    def visit_Binary(self, e: BinaryOp):
        etype = self.type_infer(e)
        if isinstance(e, (Add, Sub, Multiply, FloorDiv, Mod)) and (isinstance(etype, ScalarType) and etype.name == 'int32'):
            return self.exit_stack.enter_context(self.sb.let('v', StmtExprRewriter.visit_Binary(self, e)))
        else:
            return StmtExprRewriter.visit_Binary(self, e)

    def visit_Let(self, e: Let):
        self.exit_stack.enter_context(self.sb.let(e.var, self(e.value)))
        return self(e.body)

    def visit_Var(self, e: Var):
        return e

    def visit_Constant(self, e: Constant):
        return e

    def visit_EvaluateStmt(self, stmt: EvaluateStmt):
        with StmtContext(self):
            self.sb += StmtExprRewriter.visit_EvaluateStmt(self, stmt)

    def visit_BufferStoreStmt(self, stmt: BufferStoreStmt):
        with StmtContext(self):
            self.sb += StmtExprRewriter.visit_BufferStoreStmt(self, stmt)

    def visit_AssignStmt(self, stmt: AssignStmt):
        with StmtContext(self):
            self.sb += StmtExprRewriter.visit_AssignStmt(self, stmt)

    def visit_LetStmt(self, stmt: LetStmt):
        with StmtContext(self):
            bind_vars = stmt.bind_vars
            bind_values = [self(value) for value in stmt.bind_values]
            with self.sb.lets(bind_vars=bind_vars, values=bind_values):
                self(stmt.body)

    def visit_ForStmt(self, stmt: ForStmt):
        with StmtContext(self):
            loop_var = self.visit_expr(stmt.loop_var)
            extent = self.visit_expr(stmt.extent)
            with self.sb.for_loop(loop_var, extent, unroll=stmt.unroll):
                self.visit(stmt.body)

    def visit_IfStmt(self, stmt: IfStmt):
        with StmtContext(self):
            cond = self.visit_expr(stmt.cond)
            with self.sb.if_then(cond):
                self.visit(stmt.then_body)
            if stmt.else_body:
                with self.sb.otherwise():
                    self.visit(stmt.else_body)

    def visit_AssertStmt(self, stmt: AssertStmt):
        with StmtContext(self):
            self.sb += StmtExprRewriter.visit_AssertStmt(self, stmt)

    def visit_AsmStmt(self, stmt: AsmStmt):
        with self.exit_stack:
            input_exprs = [self.visit_expr(e) for e in stmt.input_exprs]
            output_exprs = [self.visit_expr(e) for e in stmt.output_exprs]
            self.sb += AsmStmt(stmt.template_string, list(zip(stmt.output_labels, output_exprs)),
                               list(zip(stmt.input_labels, input_exprs)), stmt.is_volatile)

    def visit_BlackBoxStmt(self, stmt: BlackBoxStmt):
        with StmtContext(self):
            exprs = [self.visit_expr(e) for e in stmt.exprs]
            self.sb += BlackBoxStmt(stmt.template_string, *exprs)

    def visit_SeqStmt(self, stmt: SeqStmt):
        for s in stmt.seq:
            self.visit(s)


class SqueezeLetStmtRewriter(StmtRewriter):
    def visit_LetStmt(self, stmt: LetStmt):
        cur = StmtRewriter.visit_LetStmt(self, stmt)

        bind_vars = []
        bind_values = []
        while isinstance(cur, LetStmt):
            bind_vars.extend(cur.bind_vars)
            bind_values.extend(cur.bind_values)
            cur = cur.body
        if same_list(bind_vars, stmt.bind_vars) and same_list(bind_values, stmt.bind_values) and cur is stmt.body:
            return stmt
        else:
            return LetStmt(bind_vars, bind_values, cur)

    def visit_SeqStmt(self, stmt: SeqStmt):
        seq = [self(s) for s in stmt.seq]
        if len(seq) == 0:
            return stmt
        body = seq[-1]
        for s in reversed(seq[:-1]):
            body = join_stmt(s, body)
        if isinstance(body, SeqStmt) and same_list(body.seq, stmt.seq):
            return stmt
        else:
            return body


def join_stmt(lhs: Stmt, rhs: Stmt):
    if isinstance(lhs, LetStmt):
        return LetStmt(lhs.bind_vars, lhs.bind_values, join_stmt(lhs.body, rhs))
    else:
        lhs_seq = lhs.seq if isinstance(lhs, SeqStmt) else [lhs]
        rhs_seq = rhs.seq if isinstance(rhs, SeqStmt) else [rhs]
        return SeqStmt(list(lhs_seq) + list(rhs_seq))


class BuildLetStmtPass(FunctionBodyPass):
    def process_body(self, stmt: Stmt) -> Stmt:
        stmt_builder = BuildLetStmtRewriter()
        squeezer = SqueezeLetStmtRewriter()
        return squeezer(stmt_builder.build(stmt))


def build_let_stmt_pass():
    return BuildLetStmtPass()
