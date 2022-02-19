from typing import Union
import functools
from collections import defaultdict
from hidet.transforms.base import RepeatFunctionPass, FunctionPass, FunctionBodyPass
from hidet.ir.functors import BoundAwareRewriter
from hidet.ir.expr import Constant, Add, Sub, Multiply, FloorDiv, Mod, Expr
from hidet.ir.stmt import LetStmt
from hidet.ir.func import Function
from hidet.ir.dialects.pattern import AnyExpr, match
from hidet.ir.functors import rewrite
from hidet.ir.analyzers import BoundAnalyzer, BoundInfo
from .bound_aware_simplify import BoundAwareSimplifier
from .rule_based_simplifier import rule_based_simplify_pass
from ...ir import Stmt


def any_expr():
    return AnyExpr(exclude_cls=Constant)


def any_constant():
    return Constant(value=None)


class TakeOutConstantRewriter(BoundAwareRewriter):
    _enumerate_limit = 1024

    def __init__(self):
        super().__init__()
        e1, e2 = any_expr(), any_expr()
        c1, c2 = any_constant(), any_constant()
        self.e1, self.e2 = e1, e2
        self.c1, self.c2 = c1, c2
        self.args = {e1, e2, c1, c2}

    def visit(self, obj):
        if obj in self.memo:
            return self.memo[obj]
        cur = BoundAwareRewriter.visit(self, obj)
        if isinstance(cur, FloorDiv):
            cur = self.post_visit_FloorDiv(cur)
        elif isinstance(obj, Mod):
            cur = self.post_visit_Mod(cur)
        self.memo[obj] = cur
        return cur

    def post_visit_FloorDiv(self, e: FloorDiv):
        e1, c1, c2 = self.e1, self.c1, self.c2
        pattern = (e1 + c1) // c2
        matching, msg = match(pattern, e)
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
        matching, msg = match(pattern, e)
        if matching:
            me1: Expr = matching[e1]
            candidates = self.bound[me1].candidate_set()
            if candidates and len(candidates) <= self._enumerate_limit:
                mc1: Constant = matching[c1]
                mc2: Constant = matching[c2]
                vc1, vc2 = mc1.value, mc2.value
                for x in candidates:
                    if (x + vc1) % vc2 != x % vc2 + vc1 % vc2:
                        break
                else:
                    # eligible to simplify
                    return me1 % mc2 + vc1 % vc2
        return e


class TakeOutConstantPass(FunctionPass):
    def __init__(self):
        super().__init__()
        self.simplifier = BoundAwareSimplifier()
        self.rewriter = TakeOutConstantRewriter()

    def process_func(self, func: Function) -> Function:
        func = self.rewriter(func)
        return self.simplifier(func)


def take_out_constant_pass():
    return RepeatFunctionPass(
        name='TakeOutConstantPass',
        passes=[
            rule_based_simplify_pass(),
            TakeOutConstantPass()
        ],
        repeat_limit=10
    )
