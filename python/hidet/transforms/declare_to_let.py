from typing import List, Dict, Sequence, Union, Tuple, Set
from collections import defaultdict
import itertools

from hidet.ir import SeqStmt, AssertStmt, IfStmt, ForTaskStmt, ForStmt, BufferStoreStmt, EvaluateStmt
from hidet.ir.expr import Var, Expr
from hidet.ir.stmt import AssignStmt, DeclareStmt, LetStmt, Stmt, BlackBoxStmt, AsmStmt, ReturnStmt
from hidet.ir.mapping import TaskMapping, SpatialTaskMapping, RepeatTaskMapping, ComposedTaskMapping
from hidet.transforms.base import Pass, FunctionBodyPass
from hidet.ir.functors import StmtFunctor, StmtExprRewriter, rewrite, collect
from hidet.utils import prod

"""
Convert DeclareStmt with initialized value to LetStmt if the declared variable has never been modified (with AssignStmt).
"""


class DeclareToLetRewriter(StmtExprRewriter):
    def __init__(self):
        super().__init__()
        self.assigns: Dict[Var, int] = defaultdict(int)

    def rewrite(self, func_body: Stmt):
        for stmt in collect(func_body, (DeclareStmt, AssignStmt, AsmStmt)):
            if isinstance(stmt, DeclareStmt):
                if stmt.init is not None:
                    self.assigns[stmt.var] += 1
            elif isinstance(stmt, AssignStmt):
                self.assigns[stmt.var] += 1
            elif isinstance(stmt, AsmStmt):
                for output_expr in stmt.output_exprs:
                    if isinstance(output_expr, Var):
                        self.assigns[output_expr] += 1
            else:
                raise ValueError()
        return self.visit(func_body)

    def visit_SeqStmt(self, seq_stmt: SeqStmt):
        seq = [self.visit(stmt) for stmt in seq_stmt.seq]
        for i in range(len(seq) - 1, -1, -1):
            stmt = seq[i]
            if isinstance(stmt, DeclareStmt):
                if self.assigns[stmt.var] == 1 and stmt.init is not None:
                    let_stmt = LetStmt(bind_vars=[stmt.var], bind_values=[stmt.init], body=self.concat(seq[i + 1:]))
                    seq = seq[:i] + [let_stmt]
        return self.concat(seq)

    def concat(self, seq: List[Stmt]):
        if len(seq) == 1:
            return seq[0]
        else:
            return SeqStmt(seq)


class DeclareToLetPass(FunctionBodyPass):
    def process_body(self, stmt: Stmt) -> Stmt:
        rewriter = DeclareToLetRewriter()
        return rewriter.rewrite(stmt)


def declare_to_let_pass() -> Pass:
    return DeclareToLetPass()
