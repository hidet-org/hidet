from hidet.ir.expr import Constant, convert
from hidet.ir.stmt import Stmt, ForStmt, SeqStmt
from hidet.ir.functors import StmtRewriter, rewrite, clone
from hidet.transforms.base import FunctionBodyPass
from hidet.transforms.expression_simplification.rule_based_simplifier import ConstExprSimplifier


class ExplicitUnrollForStmtRewriter(StmtRewriter):
    _unroll_threshold = 16

    def __init__(self):
        super().__init__()
        self.const_expr_simplifier = ConstExprSimplifier()

    def visit_ForStmt(self, stmt: ForStmt):
        extent = self.const_expr_simplifier(self.visit_expr(stmt.extent))
        body = self(stmt.body)
        if isinstance(extent, Constant) and isinstance(extent.value, int) and extent.value <= self._unroll_threshold:
            unrolled_body = []
            for i in range(extent.value):
                unrolled_body.append(clone(rewrite(body, {stmt.loop_var: convert(i)})))
            return SeqStmt(seq=unrolled_body)
        else:
            if extent is stmt.extent and body is stmt.body:
                return stmt
            else:
                return ForStmt(stmt.loop_var, extent, stmt.unroll, body)


class ExplicitUnrollForStmtPass(FunctionBodyPass):
    def process_body(self, stmt: Stmt) -> Stmt:
        rewriter = ExplicitUnrollForStmtRewriter()
        ret = rewriter(stmt)
        # print(ret)
        return ret


def explicit_unroll_for_stmt_pass():
    return ExplicitUnrollForStmtPass()
