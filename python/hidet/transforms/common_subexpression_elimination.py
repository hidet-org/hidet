from hidet.transforms.base import FunctionBodyPass, SequencePass
from hidet.ir.functors import StmtRewriter, StmtExprRewriter, same_list
from hidet.ir.expr import Expr, Var, Constant, convert
from hidet.ir.stmt import Stmt, SeqStmt, LetStmt, EvaluateStmt
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
        if same_list(new_seq, stmt.seq):
            return stmt
        else:
            return SeqStmt(new_seq)


class FlattenSeqStmtPass(FunctionBodyPass):
    def process_body(self, stmt: Stmt) -> Stmt:
        return FlattenSeqStmtRewriter()(stmt)


class ChainSeqStmtUsingLetStmtRewriter(StmtRewriter):
    def visit_SeqStmt(self, stmt: SeqStmt):
        seq = [self(s) for s in stmt.seq]
        if len(seq) == 0:
            return stmt
        body = seq[-1]
        for s in reversed(seq[:-1]):
            if isinstance(s, LetStmt):
                body = LetStmt(s.var, s.value, SeqStmt([s.body, body]))
            else:
                body = SeqStmt([s, body])
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
        var = self(stmt.var)
        value = self(stmt.value)
        if isinstance(value, (Var, Constant)):
            return self(rewrite(stmt.body, {var: value}))
        else:
            value_hash = self.expr_hash(value)
            self.value2var[value_hash] = var
            ret = self(stmt.body)
            self.value2var.pop(value_hash)
            if same_list([var, value, ret], [stmt.var, stmt.value, stmt.body]):
                return stmt
            else:
                return LetStmt(var, value, ret)


class CommonSubexpressionEliminationPass(FunctionBodyPass):
    def process_body(self, stmt: Stmt) -> Stmt:
        return CommonSubexpressionEliminationRewriter()(stmt)


def common_subexpression_elimination_pass():
    return SequencePass(name='CommonSubExpressionEliminationPassSequence',
                        passes=[
                            FlattenSeqStmtPass(),
                            ChainSeqStmtUsingLetStmtPass(),
                            CommonSubexpressionEliminationPass(),
                        ])
