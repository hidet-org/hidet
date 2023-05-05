# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import operator
from typing import Dict
from itertools import product

from hidet.ir.dialects.pattern import PlaceholderExpr, match
from hidet.ir.dtypes import boolean
from hidet.ir.expr import Add, convert, Sub, Multiply, Mod, LessThan, LessEqual, Equal, BinaryExpr, LogicalAnd
from hidet.ir.expr import BitwiseXor, BitwiseAnd, BitwiseOr, BitwiseNot
from hidet.ir.expr import Div, Constant, Expr, logical_and, logical_or, if_then_else
from hidet.ir.functors import IRRewriter
from hidet.ir.tools import rewrite
from hidet.transforms.base import FunctionPass
from hidet.utils import prod, repeat_until_converge
from hidet.ir.func import Function
from hidet.ir.analyzers import BoundAnalyzer, BoundInfo


def any_expr(allow_const):
    return PlaceholderExpr(require_non_const=not allow_const)


def any_constant():
    return PlaceholderExpr(require_const=True)


def c_div(a, b):
    if isinstance(a, int) and isinstance(b, int):
        return a // b
    else:
        return a / b


class ConstExprSimplifier(IRRewriter):
    op_dict = {
        Add: operator.add,
        Sub: operator.sub,
        Multiply: operator.mul,
        Div: c_div,
        BitwiseOr: operator.or_,
        BitwiseAnd: operator.and_,
        BitwiseXor: operator.xor,
        BitwiseNot: operator.invert,
        Mod: operator.mod,
        LessThan: operator.lt,
        LessEqual: operator.le,
        Equal: operator.eq,
    }

    def visit_Binary(self, e: BinaryExpr):
        from hidet.ir.utils.type_utils import numeric_promotion

        e = IRRewriter.visit_Binary(self, e)
        if isinstance(e.a, Constant) and isinstance(e.b, Constant) and e.__class__ in self.op_dict:
            assert isinstance(e.a, Constant) and isinstance(e.b, Constant)
            op = self.op_dict[e.__class__]
            c = op(e.a.value, e.b.value)
            if isinstance(c, bool):
                return Constant(c, 'bool')
            else:
                return Constant(c, numeric_promotion(e.a.type, e.b.type))
        return e

    def visit_And(self, e: LogicalAnd):
        e = IRRewriter.visit_Binary(self, e)
        a_val = e.a.value if isinstance(e.a, Constant) else None
        b_val = e.b.value if isinstance(e.b, Constant) else None
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


class RuleBasedSimplifier(IRRewriter):
    _enumerate_limit = 256

    def __init__(self):
        super().__init__()
        self.analyzer = BoundAnalyzer()
        self.bound: Dict[Expr, BoundInfo] = self.analyzer.bound
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
            (e1 ^ zero, e1),
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
            ((e1 - c1) - c2, e1 - (c1 + c2)),
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
            # comparison
            (e1 + c1 < c2, e1 < c2 - c1),
            (e1 - c1 < c2, e1 < c1 + c2),
            (c1 <= e1 - c2, c1 + c2 <= e1),
            (c1 <= e1 + c2, c1 - c2 <= e1),
            # and/or
            (logical_and(ec1, True), ec1),
            (logical_and(ec1, False), boolean.false),
            (logical_or(ec1, True), boolean.true),
            (logical_or(ec1, False), ec1),
            # if then else
            (if_then_else(True, ec1, ec2), ec1),
            (if_then_else(True, ec1, ec2), ec2),
        ]
        self.bound_patterns = [
            # ((pattern_args, pattern_func, target_args, target_func)
            (
                (ec1, ec2, c1),
                (ec1, ec2, c1),
                lambda ec1, ec2, c1: (ec1 + ec2) // c1,
                lambda ec1, ec2, c1: ec1 // c1 + ec2 // c1,
            ),
            (
                (ec1, ec2, c1),
                (ec1, ec2, c1),
                lambda ec1, ec2, c1: (ec1 + ec2) % c1,
                lambda ec1, ec2, c1: ec1 % c1 + ec2 % c1,
            ),
            ((ec1, c1), (ec1,), lambda ec1, c1: ec1 % c1, lambda ec1: ec1),
            ((ec1, c1, c2), (ec1, c2), lambda ec1, c1, c2: (ec1 % c1) % c2, lambda ec1, c2: ec1 % c2),
        ]

    def apply_rule(self, e):
        for pattern, target in self.patterns:
            if pattern.__class__ is not e.__class__:
                continue
            mapping, _ = match(pattern, e)
            if mapping:
                mapping = {a: b for a, b in mapping.items() if a in self.args}
                ret = rewrite(target, rewrite_map=mapping)
                return ret
        return e

    def apply_bound_aware_rule(self, e):
        for pattern_args, target_args, pattern_func, target_func in self.bound_patterns:
            pattern = pattern_func(*pattern_args)
            if pattern.__class__ is not e.__class__:
                continue
            mapping, _ = match(pattern, e)
            if mapping:
                mapping = {a: b for a, b in mapping.items() if a in self.args}
                self.analyzer(e)
                arg_candidates = {arg: self.bound[mapping[arg]].candidate_set() for arg in mapping.keys()}
                if any(can_set is None for can_set in arg_candidates.values()):
                    continue
                if prod([len(can_set) for can_set in arg_candidates.values()]) > self._enumerate_limit:
                    continue
                sorted_can_sets = []
                for pattern_arg in pattern_args:
                    sorted_can_sets.append(arg_candidates[pattern_arg])
                target_arg_index = []
                for target_arg in target_args:
                    for i in range(len(pattern_args)):
                        if pattern_args[i] is target_arg:
                            target_arg_index.append(i)
                            break
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
        self.analyzer(obj)
        if isinstance(obj, Expr):
            if obj in self.memo:
                return self.memo[obj]
            if obj in self.bound and self.bound[obj].value is not None and not isinstance(obj, Constant):
                return convert(self.bound[obj].value)
            cur = IRRewriter.visit(self, obj)
            while True:
                orig_obj = cur
                cur = self.apply_rule(cur)
                cur = self.const_expr_simplifier(cur)
                cur = self.apply_bound_aware_rule(cur)
                cur = self.const_expr_simplifier(cur)
                if orig_obj is cur:
                    break
            self.memo[obj] = cur
            return cur
        else:
            return IRRewriter.visit(self, obj)

    def visit_Mod(self, e: Mod):
        ua, ub = self.bound[e.a], self.bound[e.b]
        if ua.is_zero() or ua < ub:
            return self(e.a)
        return IRRewriter.visit_Mod(self, e)

    def visit_LessThan(self, e: LessThan):
        ua, ub = self.bound[e.a], self.bound[e.b]
        if ua < ub:
            return convert(True)
        if ub <= ua:
            return convert(False)
        return IRRewriter.visit_LessThan(self, e)

    def visit_LessEqual(self, e: LessEqual):
        ua, ub = self.bound[e.a], self.bound[e.b]
        if ua <= ub:
            return convert(True)
        if ub < ua:
            return convert(False)
        return IRRewriter.visit_LessEqual(self, e)

    def visit_Equal(self, e: Equal):
        ua, ub = self.bound[e.a], self.bound[e.b]
        if ua <= ub <= ua:
            return convert(True)
        if ua < ub or ub < ua:
            return convert(False)
        return IRRewriter.visit_Equal(self, e)

    def visit_Function(self, func: Function):
        return IRRewriter.visit_Function(self, func)


class RuleBasedSimplifyPass(FunctionPass):
    def process_func(self, func: Function) -> Function:
        simplifier = RuleBasedSimplifier()
        return repeat_until_converge(simplifier, func)


def rule_based_simplify_pass():
    return RuleBasedSimplifyPass()
