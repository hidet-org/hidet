from typing import ContextManager
from contextlib import ExitStack
from hidet.transforms.base import FunctionBodyPass, SequencePass, RepeatFunctionPass
from hidet.ir.functors import StmtRewriter, StmtExprRewriter, same_list
from hidet.ir.expr import Expr, Var, Constant, convert, Call
from hidet.ir.stmt import Stmt, SeqStmt, EvaluateStmt, IfStmt, LetStmt
from hidet.ir.func import IRModule
from hidet.ir.functors import ExprHash, rewrite
from hidet.ir.builders import FunctionBuilder, StmtBuilder


def join_stmt(lhs: Stmt, rhs: Stmt):
    if isinstance(lhs, LetStmt):
        return LetStmt(lhs.bind_vars, lhs.bind_values, join_stmt(lhs.body, rhs))
    else:
        lhs_seq = lhs.seq if isinstance(lhs, SeqStmt) else [lhs]
        rhs_seq = rhs.seq if isinstance(rhs, SeqStmt) else [rhs]
        return SeqStmt(list(lhs_seq) + list(rhs_seq))


class ChainSeqStmtUsingLetStmtRewriter(StmtRewriter):
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


class ChainSeqStmtUsingLetStmtPass(FunctionBodyPass):
    def process_body(self, stmt: Stmt) -> Stmt:
        ret = ChainSeqStmtUsingLetStmtRewriter()(stmt)
        return ret


class Value2VarContext(ContextManager):
    def __init__(self, rewriter, value_hash, var):
        self.rewriter = rewriter
        self.value_hash = value_hash
        self.var = var

    def __enter__(self):
        assert self.value_hash not in self.rewriter.value2var
        self.rewriter.value2var[self.value_hash] = self.var

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.rewriter.value2var.pop(self.value_hash)


class CommonSubexpressionEliminationRewriter(StmtExprRewriter):
    def __init__(self):
        super().__init__(use_memo=False)
        # value hash -> let var
        self.expr_hash = ExprHash()
        self.value2var = {}
        self.replace_var = {}

    def visit(self, obj):
        if isinstance(obj, Expr):
            hash_value = self.expr_hash(obj)
            if hash_value in self.value2var:
                # TODO: add a structural equivalence check, now we assume (hash(A) == hash(B) => A == B)
                return self.value2var[hash_value]
        return StmtExprRewriter.visit(self, obj)

    def visit_LetStmt(self, stmt: LetStmt):
        with ExitStack() as stack:
            bind_vars = []
            bind_values = []
            for bind_var, bind_value in zip(stmt.bind_vars, stmt.bind_values):
                updated_value = self(bind_value)
                self.expr_hash.memo[bind_var] = self.expr_hash(updated_value)
                if isinstance(updated_value, (Var, Constant)):
                    self.replace_var[bind_var] = updated_value
                else:
                    value_hash = self.expr_hash(updated_value)
                    stack.enter_context(Value2VarContext(self, value_hash, bind_var))
                    bind_vars.append(bind_var)
                    bind_values.append(updated_value)
            body = self(stmt.body)
            if same_list(bind_vars, stmt.bind_vars) and same_list(bind_values, stmt.bind_values) and body is stmt.body:
                return stmt
            else:
                if len(bind_vars) == 0:
                    return body
                else:
                    return LetStmt(bind_vars, bind_values, body)

    def visit_Var(self, e: Var):
        if e in self.replace_var:
            return self.replace_var[e]
        else:
            return e


class CommonSubexpressionEliminationPass(FunctionBodyPass):
    def process_body(self, stmt: Stmt) -> Stmt:
        return CommonSubexpressionEliminationRewriter()(stmt)


def chain_seq_stmt_using_let_stmt_pass():
    return ChainSeqStmtUsingLetStmtPass()


def common_subexpression_elimination_pass():
    return CommonSubexpressionEliminationPass()
