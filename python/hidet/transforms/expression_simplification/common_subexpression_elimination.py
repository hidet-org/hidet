from hidet.transforms.base import FunctionBodyPass, SequencePass, RepeatFunctionPass
from hidet.ir.functors import StmtRewriter, StmtExprRewriter, same_list
from hidet.ir.expr import Expr, Var, Constant, convert, Call
from hidet.ir.stmt import Stmt, SeqStmt, LetStmt, EvaluateStmt, IfStmt
from hidet.ir.func import IRModule
from hidet.ir.functors import ExprHash, rewrite
from hidet.ir.builders import FunctionBuilder, StmtBuilder


class FlattenSeqStmtRewriter(StmtRewriter):
    def visit_SeqStmt(self, stmt: SeqStmt):
        seq = [self(s) for s in stmt.seq]
        new_seq = []
        for s in seq:
            if isinstance(s, SeqStmt):
                new_seq.extend(s.seq)
            else:
                new_seq.append(s)
        if len(new_seq) == 1:
            return new_seq[0]
        elif same_list(new_seq, stmt.seq):
            return stmt
        else:
            return SeqStmt(new_seq)


class FlattenSeqStmtPass(FunctionBodyPass):
    def process_body(self, stmt: Stmt) -> Stmt:
        return FlattenSeqStmtRewriter()(stmt)


def join_stmt(lhs: Stmt, rhs: Stmt):
    if isinstance(lhs, LetStmt):
        return LetStmt(lhs.var, lhs.value, join_stmt(lhs.body, rhs))
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


class CommonSubexpressionEliminationRewriter(StmtExprRewriter):
    def __init__(self):
        super().__init__()
        # value hash -> let var
        self.expr_hash = ExprHash()
        self.value2var = {}

    def visit(self, obj):
        if isinstance(obj, Expr):
            hash_value = self.expr_hash(obj)
            if hash_value in self.value2var:
                # TODO: add a structural equivalence check, now we assume (hash(A) == hash(B) => A == B)
                self.memo[obj] = self.value2var[hash_value]
                return self.memo[obj]
        return StmtExprRewriter.visit(self, obj)

    def visit_LetStmt(self, stmt: LetStmt):
        var = stmt.var
        value = self(stmt.value)

        if isinstance(value, (Var, Constant)):
            self.expr_hash.memo[var] = self.expr_hash(value)
            self.memo[var] = value
            return self(stmt.body)
        else:
            value_hash = self.expr_hash(value)
            self.value2var[value_hash] = var
            body = self(stmt.body)
            if value_hash in self.value2var:
                self.value2var.pop(value_hash)
            if same_list([var, value, body], [stmt.var, stmt.value, stmt.body]):
                return stmt
            else:
                return LetStmt(var, value, body)


class CommonSubexpressionEliminationPass(FunctionBodyPass):
    def process_body(self, stmt: Stmt) -> Stmt:
        return CommonSubexpressionEliminationRewriter()(stmt)


def flatten_seq_stmt_pass():
    return FlattenSeqStmtPass()

def chain_seq_stmt_using_let_stmt_pass():
    return ChainSeqStmtUsingLetStmtPass()

def common_subexpression_elimination_pass():
    return CommonSubexpressionEliminationPass()
