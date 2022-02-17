import operator
from collections import defaultdict
from typing import Dict, Union

from hidet.ir.analyzers import BoundInfo, infer_bound, BoundAnalyzer
from hidet.ir.expr import Expr, Add, Sub, Multiply, Div, FloorDiv, Mod, Constant, LessThan, LessEqual, Equal, is_zero, convert
from hidet.ir.stmt import Stmt
from hidet.ir.func import Function
from hidet.ir.functors import StmtExprRewriterWithBoundAnalyzer as RewriterWithBound
from hidet.transforms import Pass, FunctionPass, FunctionBodyPass


class BoundAwareSimplifier(RewriterWithBound):
    def visit(self, obj: Union[Function, Stmt, Expr]):
        ret = RewriterWithBound.visit(self, obj)
        if obj in self.bound and not isinstance(obj, Constant) and self.bound[obj].value is not None:
            # a constant expression
            return convert(self.bound[obj].value)
        return ret

    def visit_Function(self, func: Function):
        body = self(func.body)
        if body is func.body:
            return func
        else:
            return Function(func.name, func.params, body, func.ret_type, func.local_vars, func.attrs)

    def visit_Add(self, e: Add):
        ua, ub = self.bound[e.a], self.bound[e.b]
        if ua.is_zero():
            return self(e.b)
        if ub.is_zero():
            return self(e.a)
        return RewriterWithBound.visit_Add(self, e)

    def visit_Sub(self, e: Sub):
        ua, ub = self.bound[e.a], self.bound[e.b]
        if ub.is_zero():
            return self(e.a)
        return RewriterWithBound.visit_Sub(self, e)

    def visit_Multiply(self, e: Multiply):
        ua, ub = self.bound[e.a], self.bound[e.b]
        if ua.is_one():
            return self(e.b)
        if ub.is_one():
            return self(e.a)
        if ua.is_zero() or ub.is_zero():
            return convert(0)
        return RewriterWithBound.visit_Multiply(self, e)

    def visit_Div(self, e: Div):
        ua, ub = self.bound[e.a], self.bound[e.b]
        if ua.is_zero() or ub.is_one():
            return self(e.a)
        return RewriterWithBound.visit_Div(self, e)

    def visit_Mod(self, e: Mod):
        ua, ub = self.bound[e.a], self.bound[e.b]
        if ua.is_zero() or ua < ub:
            return self(e.a)
        return RewriterWithBound.visit_Mod(self, e)

    def visit_FloorDiv(self, e: FloorDiv):
        ua, ub = self.bound[e.a], self.bound[e.b]
        if ua.is_zero() or ub.is_one():
            return self(e.a)
        if ua < ub:
            return convert(0)
        return RewriterWithBound.visit_FloorDiv(self, e)

    def visit_LessThan(self, e: LessThan):
        ua, ub = self.bound[e.a], self.bound[e.b]
        if ua < ub:
            return convert(True)
        return RewriterWithBound.visit_LessThan(self, e)

    def visit_LessEqual(self, e: LessEqual):
        ua, ub = self.bound[e.a], self.bound[e.b]
        if ua <= ub:
            return convert(True)
        return RewriterWithBound.visit_LessEqual(self, e)

    def visit_Equal(self, e: Equal):
        ua, ub = self.bound[e.a], self.bound[e.b]
        if ua <= ub <= ua:
            return convert(True)
        return RewriterWithBound.visit_Equal(self, e)


class BoundAwareSimplifyPass(FunctionBodyPass):
    def process_body(self, stmt: Stmt) -> Stmt:
        return BoundAwareSimplifier()(stmt)


def bound_aware_simplify_pass() -> Pass:
    return BoundAwareSimplifyPass()
