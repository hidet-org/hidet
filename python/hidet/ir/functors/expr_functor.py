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
# pylint: disable=bad-staticmethod-argument
from hidet.ir.expr import Add, Sub, Multiply, Div, Mod, FloorDiv, Neg, LessThan, LessEqual, Equal, NotEqual, LogicalAnd
from hidet.ir.expr import LogicalOr, LogicalNot, BitwiseAnd, BitwiseOr, BitwiseNot, BitwiseXor, LeftShift, RightShift
from hidet.ir.expr import Reference, BinaryExpr
from hidet.ir.expr import TensorElement, TensorSlice, IfThenElse, Call, Let, Var, Constant, Cast, Dereference, Address
from hidet.ir.dialects.pattern import PlaceholderExpr
from hidet.utils import same_list
from .base_functor import BaseFunctor, BaseVisitor, BaseRewriter


class ExprFunctor(BaseFunctor):
    def visit_dispatch(self, node):
        if isinstance(node, Add):
            return self.visit_Add(node)
        elif isinstance(node, Add):
            return self.visit_Add(node)
        elif isinstance(node, Sub):
            return self.visit_Sub(node)
        elif isinstance(node, Multiply):
            return self.visit_Multiply(node)
        elif isinstance(node, Div):
            return self.visit_Div(node)
        elif isinstance(node, Mod):
            return self.visit_Mod(node)
        elif isinstance(node, FloorDiv):
            return self.visit_FloorDiv(node)
        elif isinstance(node, Neg):
            return self.visit_Neg(node)
        elif isinstance(node, LessThan):
            return self.visit_LessThan(node)
        elif isinstance(node, LessEqual):
            return self.visit_LessEqual(node)
        elif isinstance(node, Equal):
            return self.visit_Equal(node)
        elif isinstance(node, NotEqual):
            return self.visit_NotEqual(node)
        elif isinstance(node, LogicalAnd):
            return self.visit_And(node)
        elif isinstance(node, LogicalOr):
            return self.visit_Or(node)
        elif isinstance(node, LogicalNot):
            return self.visit_Not(node)
        elif isinstance(node, BitwiseAnd):
            return self.visit_BitwiseAnd(node)
        elif isinstance(node, BitwiseOr):
            return self.visit_BitwiseOr(node)
        elif isinstance(node, BitwiseNot):
            return self.visit_BitwiseNot(node)
        elif isinstance(node, BitwiseXor):
            return self.visit_BitwiseXor(node)
        elif isinstance(node, LeftShift):
            return self.visit_LeftShift(node)
        elif isinstance(node, RightShift):
            return self.visit_RightShift(node)
        elif isinstance(node, TensorElement):
            return self.visit_TensorElement(node)
        elif isinstance(node, TensorSlice):
            return self.visit_TensorSlice(node)
        elif isinstance(node, IfThenElse):
            return self.visit_IfThenElse(node)
        elif isinstance(node, Call):
            return self.visit_Call(node)
        elif isinstance(node, Let):
            return self.visit_Let(node)
        elif isinstance(node, Var):
            return self.visit_Var(node)
        elif isinstance(node, Constant):
            return self.visit_Constant(node)
        elif isinstance(node, Cast):
            return self.visit_Cast(node)
        elif isinstance(node, Dereference):
            return self.visit_Dereference(node)
        elif isinstance(node, Address):
            return self.visit_Address(node)
        elif isinstance(node, Reference):
            return self.visit_Reference(node)
        elif isinstance(node, PlaceholderExpr):
            return self.visit_AnyExpr(node)
        else:
            return NotImplemented

    def visit_Add(self, e: Add):
        raise NotImplementedError()

    def visit_Sub(self, e: Sub):
        raise NotImplementedError()

    def visit_Multiply(self, e: Multiply):
        raise NotImplementedError()

    def visit_Div(self, e: Div):
        raise NotImplementedError()

    def visit_Mod(self, e: Mod):
        raise NotImplementedError()

    def visit_FloorDiv(self, e: FloorDiv):
        raise NotImplementedError()

    def visit_LessThan(self, e: LessThan):
        raise NotImplementedError()

    def visit_LessEqual(self, e: LessEqual):
        raise NotImplementedError()

    def visit_Equal(self, e: Equal):
        raise NotImplementedError()

    def visit_NotEqual(self, e: NotEqual):
        raise NotImplementedError()

    def visit_And(self, e: LogicalAnd):
        raise NotImplementedError()

    def visit_Or(self, e: LogicalOr):
        raise NotImplementedError()

    def visit_Neg(self, e: Neg):
        raise NotImplementedError()

    def visit_Not(self, e: LogicalNot):
        raise NotImplementedError()

    def visit_BitwiseAnd(self, e: BitwiseAnd):
        raise NotImplementedError()

    def visit_BitwiseOr(self, e: BitwiseOr):
        raise NotImplementedError()

    def visit_BitwiseNot(self, e: BitwiseNot):
        raise NotImplementedError()

    def visit_BitwiseXor(self, e: BitwiseXor):
        raise NotImplementedError()

    def visit_LeftShift(self, e: LeftShift):
        raise NotImplementedError()

    def visit_RightShift(self, e: RightShift):
        raise NotImplementedError()

    def visit_TensorElement(self, e: TensorElement):
        raise NotImplementedError()

    def visit_TensorSlice(self, e: TensorSlice):
        raise NotImplementedError()

    def visit_IfThenElse(self, e: IfThenElse):
        raise NotImplementedError()

    def visit_Cast(self, e: Cast):
        raise NotImplementedError()

    def visit_Dereference(self, e: Dereference):
        raise NotImplementedError()

    def visit_Address(self, e: Address):
        raise NotImplementedError()

    def visit_Reference(self, e: Reference):
        raise NotImplementedError()

    def visit_Call(self, e: Call):
        raise NotImplementedError()

    def visit_Let(self, e: Let):
        raise NotImplementedError()

    def visit_Var(self, e: Var):
        raise NotImplementedError()

    def visit_Constant(self, e: Constant):
        raise NotImplementedError()

    def visit_AnyExpr(self, e: PlaceholderExpr):
        raise NotImplementedError()


class ExprVisitor(ExprFunctor, BaseVisitor):
    def visit_Add(self, e: Add):
        self.visit(e.a)
        self.visit(e.b)

    def visit_Sub(self, e: Sub):
        self.visit(e.a)
        self.visit(e.b)

    def visit_Multiply(self, e: Multiply):
        self.visit(e.a)
        self.visit(e.b)

    def visit_Div(self, e: Div):
        self.visit(e.a)
        self.visit(e.b)

    def visit_Mod(self, e: Mod):
        self.visit(e.a)
        self.visit(e.b)

    def visit_FloorDiv(self, e: FloorDiv):
        self.visit(e.a)
        self.visit(e.b)

    def visit_LessThan(self, e: LessThan):
        self.visit(e.a)
        self.visit(e.b)

    def visit_LessEqual(self, e: LessEqual):
        self.visit(e.a)
        self.visit(e.b)

    def visit_Equal(self, e: Equal):
        self.visit(e.a)
        self.visit(e.b)

    def visit_NotEqual(self, e: NotEqual):
        self.visit(e.a)
        self.visit(e.b)

    def visit_And(self, e: LogicalAnd):
        self.visit(e.a)
        self.visit(e.b)

    def visit_Or(self, e: LogicalOr):
        self.visit(e.a)
        self.visit(e.b)

    def visit_Neg(self, e: Neg):
        self.visit(e.a)

    def visit_Not(self, e: LogicalNot):
        self.visit(e.a)

    def visit_BitwiseAnd(self, e: BitwiseAnd):
        self.visit(e.a)
        self.visit(e.b)

    def visit_BitwiseOr(self, e: BitwiseOr):
        self.visit(e.a)
        self.visit(e.b)

    def visit_BitwiseXor(self, e: BitwiseXor):
        self.visit(e.a)
        self.visit(e.b)

    def visit_BitwiseNot(self, e: BitwiseNot):
        self.visit(e.a)

    def visit_LeftShift(self, e: LeftShift):
        self.visit(e.a)
        self.visit(e.b)

    def visit_RightShift(self, e: RightShift):
        self.visit(e.a)
        self.visit(e.b)

    def visit_TensorElement(self, e: TensorElement):
        self.visit(e.base)
        for idx in e.indices:
            self.visit(idx)

    def visit_TensorSlice(self, e: TensorSlice):
        self.visit(e.base)
        for idx, start, end in zip(e.starts, e.indices, e.ends):
            for obj in [idx, start, end]:
                if obj is not None:
                    self.visit(obj)

    def visit_IfThenElse(self, e: IfThenElse):
        self.visit(e.cond)
        self.visit(e.then_expr)
        self.visit(e.else_expr)

    def visit_Call(self, e: Call):
        self.visit(e.func_var)
        for arg in e.args:
            self.visit(arg)

    def visit_Let(self, e: Let):
        self.visit(e.value)
        self.visit(e.var)
        self.visit(e.body)

    def visit_Var(self, e: Var):
        pass

    def visit_Constant(self, e: Constant):
        pass

    # low level dialect
    def visit_Cast(self, e: Cast):
        self.visit(e.expr)

    def visit_Dereference(self, e: Dereference):
        self.visit(e.expr)

    def visit_Address(self, e: Address):
        self.visit(e.expr)

    def visit_Reference(self, e: Reference):
        self.visit(e.expr)

    def visit_AnyExpr(self, e: PlaceholderExpr):
        pass


class ExprRewriter(ExprFunctor, BaseRewriter):
    def rewrite(self, e):
        return self.visit(e)

    def visit_Binary(self, e: BinaryExpr):
        a = self(e.a)
        b = self(e.b)
        if a is e.a and b is e.b:
            return e
        else:
            return e.__class__(a, b)

    def visit_Add(self, e: Add):
        return self.visit_Binary(e)

    def visit_Sub(self, e: Sub):
        return self.visit_Binary(e)

    def visit_Multiply(self, e: Multiply):
        return self.visit_Binary(e)

    def visit_Div(self, e: Div):
        return self.visit_Binary(e)

    def visit_Mod(self, e: Mod):
        return self.visit_Binary(e)

    def visit_FloorDiv(self, e: FloorDiv):
        return self.visit_Binary(e)

    def visit_LessThan(self, e: LessThan):
        return self.visit_Binary(e)

    def visit_LessEqual(self, e: LessEqual):
        return self.visit_Binary(e)

    def visit_Equal(self, e: Equal):
        return self.visit_Binary(e)

    def visit_NotEqual(self, e: NotEqual):
        return self.visit_Binary(e)

    def visit_And(self, e: LogicalAnd):
        return self.visit_Binary(e)

    def visit_Or(self, e: LogicalOr):
        return self.visit_Binary(e)

    def visit_Neg(self, e: Neg):
        a = self(e.a)
        if a is e.a:
            return e
        else:
            return Neg(a)

    def visit_Not(self, e: LogicalNot):
        a = self(e.a)
        if a is e.a:
            return e
        else:
            return LogicalNot(a)

    def visit_BitwiseAnd(self, e: BitwiseAnd):
        return self.visit_Binary(e)

    def visit_BitwiseOr(self, e: BitwiseOr):
        return self.visit_Binary(e)

    def visit_BitwiseXor(self, e: BitwiseXor):
        return self.visit_Binary(e)

    def visit_BitwiseNot(self, e: BitwiseNot):
        base = self.visit(e.a)
        if base is e.a:
            return e
        else:
            return BitwiseNot(base)

    def visit_LeftShift(self, e: LeftShift):
        base = self.visit(e.a)
        cnt = self.visit(e.b)
        if base is e.a and cnt is e.b:
            return e
        else:
            return LeftShift(base, cnt)

    def visit_RightShift(self, e: RightShift):
        base = self.visit(e.a)
        cnt = self.visit(e.b)
        if base is e.a and cnt is e.b:
            return e
        else:
            return RightShift(base, cnt)

    def visit_TensorElement(self, e: TensorElement):
        base = self(e.base)
        indices = tuple(self(idx) if idx is not None else None for idx in e.indices)
        if base is e.base and same_list(indices, e.indices):
            return e
        else:
            return TensorElement(base, indices, e.protected)

    def visit_TensorSlice(self, e: TensorSlice):
        base = self(e.base)
        indices = [self(idx) if idx is not None else None for idx in e.indices]
        starts = [self(start) if start is not None else None for start in e.starts]
        ends = [self(end) if end is not None else None for end in e.ends]
        if base is e.base and same_list(indices, e.indices) and same_list(starts, e.starts) and same_list(ends, e.ends):
            return e
        else:
            return TensorSlice(base, indices, starts, ends)

    def visit_IfThenElse(self, e: IfThenElse):
        cond = self(e.cond)
        then_expr = self(e.then_expr)
        else_expr = self(e.else_expr)
        if cond is e.cond and then_expr is e.then_expr and else_expr is e.else_expr:
            return e
        else:
            return IfThenElse(cond, then_expr, else_expr)

    def visit_Cast(self, e: Cast):
        expr = self(e.expr)
        if expr is e.expr:
            return e
        else:
            return Cast(expr, e.target_type)

    def visit_Dereference(self, e: Dereference):
        expr = self(e.expr)
        if expr is e.expr:
            return e
        else:
            return Dereference(expr)

    def visit_Address(self, e: Address):
        expr = self(e.expr)
        if expr is e.expr:
            return e
        else:
            return Address(expr)

    def visit_Reference(self, e: Reference):
        expr = self(e.expr)
        if expr is e.expr:
            return e
        else:
            return Reference(expr)

    def visit_Call(self, e: Call):
        func_var = self(e.func_var)
        args = tuple(self(arg) for arg in e.args)
        if func_var is e.func_var and same_list(args, e.args):
            return e
        else:
            return Call(func_var, args)

    def visit_Let(self, e: Let):
        v = e.var
        value = self(e.value)
        body = self(e.body)
        if same_list([v, value, body], [e.var, e.value, e.body]):
            return e
        else:
            return Let(v, value, body)

    def visit_Var(self, e: Var):
        return e

    def visit_Constant(self, e: Constant):
        return e

    def visit_AnyExpr(self, e: PlaceholderExpr):
        return e
