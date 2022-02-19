import operator
from itertools import product

from hidet.ir.dialects.pattern import AnyExpr, match
from hidet.ir.expr import Add, convert, Sub, Multiply, FloorDiv, Mod, LessThan, LessEqual, Equal, BinaryOp, And, IfThenElse
from hidet.ir.expr import Constant, Expr
from hidet.ir.functors import BoundAwareRewriter
from hidet.ir.functors import StmtExprRewriter
from hidet.ir.functors import rewrite, ExprHash
from hidet.transforms.base import FunctionPass
from hidet.utils import prod, repeat_until_converge
from hidet.utils.py import DictCustomKey
from hidet.ir.func import Function


def any_expr(allow_const):
    if allow_const:
        return AnyExpr()
    else:
        return AnyExpr(exclude_cls=Constant)


def any_constant():
    return Constant(value=None)


class ConstExprSimplifier(StmtExprRewriter):
    op_dict = {
        Add: operator.add,
        Sub: operator.sub,
        Multiply: operator.mul,
        FloorDiv: operator.floordiv,
        Mod: operator.mod,
        LessThan: operator.lt,
        LessEqual: operator.le,
        Equal: operator.eq,
    }

    def visit_Binary(self, e: BinaryOp):
        e = StmtExprRewriter.visit_Binary(self, e)
        if e.a.is_const() and e.b.is_const() and e.__class__ in self.op_dict:
            op = self.op_dict[e.__class__]
            return convert(op(e.a.const().value, e.b.const().value))
        return e

    def visit_And(self, e: And):
        e = StmtExprRewriter.visit_Binary(self, e)
        a_val = e.a.const().value if e.a.is_const() else None
        b_val = e.b.const().value if e.b.is_const() else None
        if a_val and b_val:
            return convert(True)
        elif a_val is False or b_val is False:
            return convert(False)
        elif a_val:
            return e.b
        elif b_val:
            return e.a
        else:
            return e


class RuleBasedSimplifier(BoundAwareRewriter):
    _enumerate_limit = 128

    def __init__(self):
        super().__init__()
        self.memo = DictCustomKey(hash_func=ExprHash())
        self.const_expr_simplifier = ConstExprSimplifier()
        e1, e2 = any_expr(allow_const=False), any_expr(allow_const=False)
        c1, c2 = any_constant(), any_constant()
        ec1, ec2 = any_expr(allow_const=True), any_expr(allow_const=True)
        zero = convert(0)
        one = convert(1)
        self.args = {e1, e2, c1, c2, ec1, ec2}
        self.patterns = [
            (e1 + zero, e1),
            (e1 - zero, e1),
            (e1 * one, e1),
            (e1 * zero, zero),
            (e1 // one, e1),
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
            ((e1 % c1) % c1, e1 % c1),
            # if then else
            (IfThenElse(True, ec1, ec2), ec1),
            (IfThenElse(False, ec1, ec2), ec2),
        ]
        self.bound_patterns = [
            # ((pattern_args, pattern_func, target_args, target_func)
            ((e1, c1, c2), (e1, c1, c2), lambda e1, c1, c2: (e1 + c1) // c2, lambda e1, c1, c2: e1 // c2 + c1 // c2),
            ((e1, c1, c2), (e1, c1, c2), lambda e1, c1, c2: (e1 + c1) % c2, lambda e1, c1, c2: e1 % c2 + c1 % c2),
            ((e1, e2, c1), (e1, e2, c1), lambda e1, e2, c1: (e1 + e2) // c1, lambda e1, e2, c1: e1 // c1 + e2 // c1),
            ((e1, e2, c1), (e1, e2, c1), lambda e1, e2, c1: (e1 + e2) % c1, lambda e1, e2, c1: e1 % c1 + e2 % c1),
            ((e1, c1), (e1,), lambda e1, c1: e1 % c1, lambda e1: e1),
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

    def apply_bound_aware_rule(self, e):
        for idx, (pattern_args, target_args, pattern_func, target_func) in enumerate(self.bound_patterns):
            pattern = pattern_func(*pattern_args)
            if pattern.__class__ is not e.__class__:
                continue
            mapping, msg = match(pattern, e)
            if mapping:
                mapping = {a: b for a, b in mapping.items() if a in self.args}
                self.analyzer(e)
                arg_candidates = {arg: self.bound[mapping[arg]].candidate_set() for arg in mapping.keys()}
                if any(can_set is None for can_set in arg_candidates.values()):
                    continue
                if prod([len(can_set) for can_set in arg_candidates.values()]) > self._enumerate_limit:
                    continue
                sorted_can_sets = [can_set for arg, can_set in sorted(arg_candidates.items(), key=lambda args: pattern_args.index(args[0]))]
                target_arg_index = [pattern_args.index(arg) for arg in target_args]
                for args in product(*sorted_can_sets):
                    t_args = [args[i] for i in target_arg_index]
                    if pattern_func(*args) != target_func(*t_args):
                        break
                else:
                    target = target_func(*target_args)
                    ret = rewrite(target, rewrite_map=mapping)
                    return ret
        return e

    def visit(self, obj):
        if obj in self.memo:
            return self.memo[obj]
        cur = BoundAwareRewriter.visit(self, obj)
        if isinstance(cur, Expr):
            while True:
                orig_obj = cur
                cur = self.apply_rule(cur)
                cur = self.apply_bound_aware_rule(cur)
                if orig_obj is cur:
                    break
        self.memo[obj] = cur
        return cur

    def visit_Mod(self, e: Mod):
        ua, ub = self.bound[e.a], self.bound[e.b]
        if ua.is_zero() or ua < ub:
            return self(e.a)
        return BoundAwareRewriter.visit_Mod(self, e)

    def visit_LessThan(self, e: LessThan):
        ua, ub = self.bound[e.a], self.bound[e.b]
        if ua < ub:
            return convert(True)
        return BoundAwareRewriter.visit_LessThan(self, e)

    def visit_LessEqual(self, e: LessEqual):
        ua, ub = self.bound[e.a], self.bound[e.b]
        if ua <= ub:
            return convert(True)
        return BoundAwareRewriter.visit_LessEqual(self, e)

    def visit_Equal(self, e: Equal):
        ua, ub = self.bound[e.a], self.bound[e.b]
        if ua <= ub <= ua:
            return convert(True)
        return BoundAwareRewriter.visit_Equal(self, e)


class RuleBasedSimplifyPass(FunctionPass):
    def process_func(self, func: Function) -> Function:
        return repeat_until_converge(RuleBasedSimplifier(), func)


def rule_based_simplify_pass():
    return RuleBasedSimplifyPass()
