from hidet.ir.dialects.pattern import AnyExpr, match
from hidet.ir.expr import Constant
from hidet.ir.functors import StmtExprRewriter
from hidet.ir.functors import rewrite, ExprHash
from hidet.ir.stmt import Stmt
from hidet.transforms.base import FunctionBodyPass
from hidet.utils.py import DictCustomKey
from .const_expr_simplifier import ConstExprSimplifier


def any_expr():
    return AnyExpr(exclude_cls=Constant)


def any_constant():
    return Constant(value=None)


class RuleBasedSimplifier(StmtExprRewriter):
    def __init__(self):
        super().__init__()
        self.memo = DictCustomKey(hash_func=ExprHash())
        self.const_expr_simplifier = ConstExprSimplifier()
        e1, e2 = any_expr(), any_expr()
        c1, c2 = any_constant(), any_constant()
        self.args = {e1, e2, c1, c2}
        self.patterns = [
            # add
            ((c1 + e1) + e2, (e1 + e2) + c1),
            ((e1 + c1) + c2, e1 + (c1 + c2)),
            ((c1 - e1) + e2, (e2 - e1) + c1),
            ((e1 - c1) + e2, (e1 + e2) - c1),
            # sub
            ((c1 + e1) - e2, (e1 - e2) + c1),
            (e1 - (c1 + e2), (e1 - e2) - c1),
            ((c1 - e1) - e2, c1 - (e1 + e2)),
            ((e1 - c1) - e2, (e1 - e2) - c1),
            (e1 - (c1 - e2), (e1 + e2) - c1),
            (e1 - (e2 - c1), (e1 - e2) + c1),
            # mul
            ((e1 + c1) * c2, c1 * c2 + e1 * c2),
            ((c1 - e1) * c2, c1 * c2 - e1 * c2),
            ((e1 - c1) * c2, e1 * c2 - c1 * c2),
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
                mapping = {a: b for a, b in mapping.items() if a in self.args}
                ret = rewrite(target, rewrite_map=mapping)
                ret = self.const_expr_simplifier(ret)
                return ret
        return e

    def visit(self, obj):
        if obj in self.memo:
            return self.memo[obj]
        cur = obj
        while True:
            orig_obj = cur
            cur = StmtExprRewriter.visit(self, cur)
            cur = self.apply_rule(cur)
            if orig_obj is cur:
                break
        self.memo[obj] = cur
        return cur


class RuleBasedSimplifyPass(FunctionBodyPass):
    def process_body(self, stmt: Stmt) -> Stmt:
        simplifier = RuleBasedSimplifier()
        return simplifier(stmt)


def rule_based_simplify_pass():
    return RuleBasedSimplifyPass()
