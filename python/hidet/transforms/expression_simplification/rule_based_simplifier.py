from collections import defaultdict
from typing import Union

from hidet.ir.dialects.pattern import AnyExpr, match
from hidet.ir.expr import Constant, Add, Sub, Multiply, FloorDiv, Mod, Expr
from hidet.ir.func import Function
from hidet.ir.functors import StmtExprRewriter
from hidet.ir.functors import rewrite
from hidet.ir.stmt import LetStmt
from hidet.transforms.base import RepeatFunctionPass, FunctionPass, FunctionBodyPass
from .bound_aware_simplify import BoundAwareSimplifier
from .const_expr_simplifier import ConstExprSimplifier
from ...ir import Stmt


def any_expr():
    return AnyExpr(exclude_cls=Constant)


def any_constant():
    return Constant(value=None)


class RuleBasedSimplifier(StmtExprRewriter):
    def __init__(self):
        super().__init__()
        e1, e2 = any_expr(), any_expr()
        c1, c2 = any_constant(), any_constant()
        self.e1, self.e2 = e1, e2
        self.c1, self.c2 = c1, c2
        self.args = {e1, e2, c1, c2}
        self.var2value = {}
        self.patterns = [
            # add
            ((c1 + e1) + e2, (e1 + e2) + c1),
            ((e1 + c1) + e2, (e1 + e2) + c1),
            ((e1 + c1) + c2, e1 + (c1 + c2)),
            ((c1 + e1) + c2, e1 + (c1 + c2)),
            (e1 + (c1 + e2), (e1 + e2) + c1),
            (e1 + (e2 + c1), (e1 + e2) + c1),
            ((c1 - e1) + e2, (e2 - e1) + c1),
            ((e1 - c1) + e2, (e1 + e2) - c1),
            (e1 + (c1 - e2), (e1 - e2) + c1),
            (e1 + (e2 - c1), (e1 + e2) - c1),
            # sub
            ((c1 + e1) - e2, (e1 - e2) + c1),
            ((e1 + c1) - e2, (e1 - e2) + c1),
            (e1 - (c1 + e2), (e1 - e2) - c1),
            (e1 - (e2 + c1), (e1 - e2) - c1),
            ((c1 - e1) - e2, c1 - (e1 + e2)),
            ((e1 - c1) - e2, (e1 - e2) - c1),
            (e1 - (c1 - e2), (e1 + e2) - c1),
            (e1 - (e2 - c1), (e1 - e2) + c1),
            # mul
            ((c1 + e1) * c2, c1 * c2 + e1 * c2),
            ((e1 + c1) * c2, e1 * c2 + c1 * c2),
            (c1 * (c2 + e1), c1 * c2 + c1 * e1),
            (c1 * (e1 + c2), c1 * e1 + c1 * c2),
            ((c1 - e1) * c2, c1 * c2 - e1 * c2),
            ((e1 - c1) * c2, e1 * c2 - c1 * c2),
            (c1 * (c2 - e1), c1 * c2 - c1 * e1),
            (c1 * (e1 - c2), c1 * e1 - c1 * c2),
            ((e1 * c1) * c2, e1 * (c1 * c2)),
            # div
            (((e1 * c1) + (e2 % c1)) // c1, e1),
            ((e1 // c1) // c2, e1 // (c1 * c2)),
            ((e1 * c1) // c1, e1),
            ((e1 * c1 + e2) // c1, e1 + e2 // c1),
            # mod
            ((e1 * c1 + e2) % c1, e2 % c1),
            ((e1 % c1) % c1, e1 % c1)
        ]

    def apply_rule(self, e):
        for idx, (pattern, target) in enumerate(self.patterns):
            if pattern.__class__ is not e.__class__:
                continue
            mapping, msg = match(pattern, e)
            if mapping:
                # print('apply rule ', pattern, target, 'on', e)
                # print('applying rule ', idx + 1)
                mapping = {a: b for a, b in mapping.items() if a in self.args}
                return rewrite(target, rewrite_map=mapping)
        return e

    def visit(self, obj):
        if obj in self.memo:
            return self.memo[obj]
        cur = StmtExprRewriter.visit(self, obj)
        while True:
            orig_obj = cur
            cur = self.apply_rule(cur)
            if orig_obj is cur:
                break
        self.memo[obj] = cur
        return cur


class RuleBasedSimplifyPass(FunctionBodyPass):
    def process_body(self, stmt: Stmt) -> Stmt:
        sa, sb = RuleBasedSimplifier(), ConstExprSimplifier()
        # print(stmt)
        while True:
            orig_stmt = stmt
            stmt = sb(sa(stmt))
            # print(stmt)
            if orig_stmt is stmt:
                break
        return stmt


def rule_based_simplify_pass():
    return RuleBasedSimplifyPass()
