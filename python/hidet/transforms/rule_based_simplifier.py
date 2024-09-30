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

from hidet.ir import dtypes
from hidet.ir.dialects.pattern import PlaceholderExpr, match
from hidet.ir.dtypes import boolean, int32
from hidet.ir.expr import Add, convert, Sub, Multiply, Mod, LessThan, LessEqual, Equal, BinaryExpr, LogicalAnd
from hidet.ir.expr import BitwiseXor, BitwiseAnd, BitwiseOr, BitwiseNot, Var, LogicalOr
from hidet.ir.expr import Div, Constant, Expr, logical_and, constant, IfThenElse
from hidet.ir.stmt import LetStmt, ForStmt
from hidet.ir.functors import IRRewriter
from hidet.ir.tools import rewrite, simplify
from hidet.transforms.base import FunctionPass
from hidet.utils import prod, repeat_until_converge, same_list
from hidet.ir.func import Function
from hidet.ir.analyzers import BoundAnalyzer, BoundInfo


def any_expr(allow_const):
    return PlaceholderExpr(require_non_const=not allow_const)


def any_constant():
    return PlaceholderExpr(require_const=True)


def int_constant():
    return PlaceholderExpr(required_type=int32, require_const=True)


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
                return constant(c, 'bool')
            else:
                return constant(c, numeric_promotion(e.a.type, e.b.type))
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
        ic1, ic2 = int_constant(), int_constant()
        ec1, ec2 = any_expr(allow_const=True), any_expr(allow_const=True)
        self.args = {e1, e2, c1, c2, ic1, ic2, ec1, ec2}
        true = constant(True, boolean)
        false = constant(False, boolean)
        self.patterns = [
            # add
            ((c1 + e1) + e2, (e1 + e2) + c1),
            ((e1 + c1) + c2, e1 + (c1 + c2)),
            ((c1 - e1) + e2, (e2 - e1) + c1),
            ((e1 - c1) + e2, (e1 + e2) - c1),
            # sub
            (ec1 - ec1, dtypes.int32.zero),
            ((c1 + e1) - e2, (e1 - e2) + c1),
            (e1 - (c1 + e2), (e1 - e2) - c1),
            ((c1 - e1) - e2, c1 - (e1 + e2)),
            ((e1 - c1) - e2, (e1 - e2) - c1),
            (e1 - (c1 - e2), (e1 + e2) - c1),
            (e1 - (e2 - c1), (e1 - e2) + c1),
            ((e1 - c1) - c2, e1 - (c1 + c2)),
            # mul
            # for the following three pattern rules, the third item is the condition to apply the pattern rules.
            # they are used to avoid arithmetic overflow in cases like (1-(j <= i ? 1.0 : 0.0)) * (-1e9).
            # because 1 + 1e9 - 1e9 becomes 0 for float32 data type
            ((e1 + c1) * c2, c1 * c2 + e1 * c2, logical_and(c2 <= 1e5, -c2 <= 1e5)),
            ((c1 - e1) * c2, c1 * c2 - e1 * c2, logical_and(c2 <= 1e5, -c2 <= 1e5)),
            ((e1 - c1) * c2, e1 * c2 - c1 * c2, logical_and(c2 <= 1e5, -c2 <= 1e5)),
            ((e1 * c1) * c2, e1 * (c1 * c2)),
            # div
            (ec1 // ec1, dtypes.int32.one),
            (((e1 * c1) + (e2 % c1)) // c1, e1),
            ((e1 // ic1) // ic2, e1 // (ic1 * ic2)),
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
            (LogicalAnd(ec1, true), ec1),
            (LogicalAnd(ec1, false), false),
            (LogicalOr(ec1, true), true),
            (LogicalOr(ec1, false), ec1),
            # if then else
            (IfThenElse(true, ec1, ec2), ec1),
            (IfThenElse(false, ec1, ec2), ec2),
            (IfThenElse(ec1, ec2, ec2), ec2),
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
        for items in self.patterns:
            if len(items) == 2:
                pattern, target = items
                condition = boolean.true
            else:
                pattern, target, condition = items
            if pattern.__class__ is not e.__class__:
                continue
            mapping, _ = match(pattern, e)
            if mapping:
                condition = simplify(rewrite(condition, rewrite_map=mapping))
                assert isinstance(condition, Constant)
                if bool(condition):
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

    def visit_LetStmt(self, stmt: LetStmt):
        bind_vars = self(stmt.bind_vars)
        bind_values = [self.visit(bind_value) for bind_value in stmt.bind_values]
        body = self.visit(stmt.body)
        bind_vars = [
            updated if isinstance(updated, Var) else original for original, updated in zip(stmt.bind_vars, bind_vars)
        ]
        if same_list(bind_vars, stmt.bind_vars) and same_list(bind_values, stmt.bind_values) and body is stmt.body:
            return stmt
        else:
            return LetStmt(bind_vars, bind_values, body)

    def visit_ForStmt(self, stmt: ForStmt):
        loop_var = self(stmt.loop_var)
        loop_var = loop_var if isinstance(loop_var, Var) else stmt.loop_var
        extent = self.visit(stmt.extent)
        body = self.visit(stmt.body)
        if loop_var is stmt.loop_var and extent is stmt.extent and body is stmt.body:
            return stmt
        else:
            return ForStmt(loop_var, extent, body=body, attr=stmt.attr)

    def visit_Function(self, func: Function):
        return IRRewriter.visit_Function(self, func)


class RuleBasedSimplifyPass(FunctionPass):
    def process_func(self, func: Function) -> Function:
        simplifier = RuleBasedSimplifier()
        return repeat_until_converge(simplifier, func)


def rule_based_simplify_pass():
    return RuleBasedSimplifyPass()
