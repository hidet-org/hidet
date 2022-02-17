from typing import Union
import functools
from hidet.transforms.base import RepeatFunctionPass, FunctionPass, FunctionBodyPass
from hidet.ir.functors import StmtExprRewriterWithBoundAnalyzer
from hidet.ir.expr import Constant, Add, Sub, Multiply, FloorDiv, Mod, Expr
from hidet.ir.stmt import LetStmt
from hidet.ir.func import Function
from hidet.ir.dialects.pattern import AnyExpr, match
from hidet.ir.functors import rewrite
from hidet.ir.analyzers import BoundAnalyzer, BoundInfo
from .bound_aware_simplify import BoundAwareSimplifier
from ...ir import Stmt


def any_expr():
    return AnyExpr(exclude_cls=Constant)


def any_constant():
    return Constant(value=None)


class TakeOutConstantRewriter(StmtExprRewriterWithBoundAnalyzer):
    _enumerate_limit = 1024

    def __init__(self):
        super().__init__()
        e1, e2 = any_expr(), any_expr()
        c1, c2 = any_constant(), any_constant()
        self.e1, self.e2 = e1, e2
        self.c1, self.c2 = c1, c2
        self.args = {e1, e2, c1, c2}
        self.var2value = {}
        self.patterns = {
            Add: [
                ((c1 + e1) + e2, (e1 + e2) + c1),
                ((e1 + c1) + e2, (e1 + e2) + c1),
                (e1 + (c1 + e2), (e1 + e2) + c1),
                (e1 + (e2 + c1), (e1 + e2) + c1),
                ((c1 - e1) + e2, (e2 - e1) + c1),
                ((e1 - c1) + e2, (e1 + e2) - c1),
                (e1 + (c1 - e2), (e1 - e2) + c1),
                (e1 + (e2 - c1), (e1 + e2) - c1),
            ],
            Sub: [
                ((c1 + e1) - e2, (e1 - e2) + c1),
                ((e1 + c1) - e2, (e1 - e2) + c1),
                (e1 - (c1 + e2), (e1 - e2) - c1),
                (e1 - (e2 + c1), (e1 - e2) - c1),
                ((c1 - e1) - e2, c1 - (e1 + e2)),
                ((e1 - c1) - e2, (e1 - e2) - c1),
                (e1 - (c1 - e2), (e1 + e2) - c1),
                (e1 - (e2 - c1), (e1 - e2) + c1),
            ],
            Multiply: [
                ((c1 + e1) * c2, c1 * c2 + e1 * c2),
                ((e1 + c1) * c2, e1 * c2 + c1 * c2),
                (c1 * (c2 + e1), c1 * c2 + c1 * e1),
                (c1 * (e1 + c2), c1 * e1 + c1 * c2),
                ((c1 - e1) * c2, c1 * c2 - e1 * c2),
                ((e1 - c1) * c2, e1 * c2 - c1 * c2),
                (c1 * (c2 - e1), c1 * c2 - c1 * e1),
                (c1 * (e1 - c2), c1 * e1 - c1 * c2),
            ],
        }

    def visit(self, obj):
        if obj in self.memo:
            return self.memo[obj]
        ret = StmtExprRewriterWithBoundAnalyzer.visit(self, obj)
        if isinstance(ret, (Add, Sub, Multiply)):
            ret = self.post_visit_Add_Sub_Multiply(ret, self.patterns[ret.__class__])
        elif isinstance(ret, FloorDiv):
            ret = self.post_visit_FloorDiv(ret)
        elif isinstance(ret, Mod):
            ret = self.post_visit_Mod(ret)
        self.memo[obj] = ret
        return ret

    def post_visit_Add_Sub_Multiply(self, e: Union[Add, Sub, Multiply], patterns):
        for pattern, target in patterns:
            mapping, msg = match(pattern, e)
            if mapping:
                mapping = {a: b for a, b in mapping.items() if a in self.args}
                return rewrite(target, rewrite_map=mapping)
        return e

    def post_visit_FloorDiv(self, e: FloorDiv):
        e1, c1, c2 = self.e1, self.c1, self.c2
        pattern = (e1 + c1) // c2
        if e.a in self.var2value:
            target = FloorDiv(self.var2value[e.a], e.b)
        else:
            target = e
        matching, msg = match(pattern, target)
        if matching:
            me1: Expr = matching[e1]
            candidates = self.bound[me1].candidate_set()
            if candidates and len(candidates) <= self._enumerate_limit:
                mc1: Constant = matching[c1]
                mc2: Constant = matching[c2]
                vc1, vc2 = mc1.value, mc2.value
                for x in candidates:
                    if (x + vc1) // vc2 != x // vc2:
                        break
                else:
                    # eligible to simplify
                    return me1 // mc2
        return e

    def post_visit_Mod(self, e: Mod):
        e1, c1, c2 = self.e1, self.c1, self.c2
        pattern = (e1 + c1) % c2
        if e.a in self.var2value:
            target = FloorDiv(self.var2value[e.a], e.b)
        else:
            target = e
        matching, msg = match(pattern, target)
        if matching:
            me1: Expr = matching[e1]
            candidates = self.bound[me1].candidate_set()
            if candidates and len(candidates) <= self._enumerate_limit:
                mc1: Constant = matching[c1]
                mc2: Constant = matching[c2]
                vc1, vc2 = mc1.value, mc2.value
                for x in candidates:
                    if (x + vc1) % vc2 != x % vc2:
                        break
                else:
                    # eligible to simplify
                    return me1 % mc2
        return e

    def visit_LetStmt(self, stmt: LetStmt):
        var = self.visit(stmt.var)
        value = self.visit(stmt.value)
        self.var2value[var] = value
        return StmtExprRewriterWithBoundAnalyzer.visit_LetStmt(self, stmt)


class TakeOutConstantPass(FunctionBodyPass):
    def __init__(self):
        super().__init__()
        self.simplifier = BoundAwareSimplifier()
        self.rewriter = TakeOutConstantRewriter()

    def process_body(self, stmt: Stmt) -> Stmt:
        ret = self.simplifier(self.rewriter(stmt))
        return ret


def take_out_constant_pass():
    return RepeatFunctionPass(
        name='TakeOutConstantPass',
        passes=[
            TakeOutConstantPass()
        ],
        repeat_limit=10
    )
