import operator
from hidet.core.compute import ReduceCompute, TensorCompute, TensorInput, ScalarInput
from hidet.ir.expr import *
from hidet.ir.functors import ExprFunctor, same


class Simplifier(ExprFunctor):
    def visit_Binary(self, e: BinaryOp):
        a = self(e.a)
        b = self(e.b)
        if isinstance(e, Add):
            if is_zero(a):
                return b
            if is_zero(b):
                return a
        elif isinstance(e, Sub):
            if is_zero(b):
                return a
        elif isinstance(e, Multiply):
            if is_one(a):
                return b
            if is_one(b):
                return a
        elif isinstance(e, Div):
            if is_one(b):
                return a
        elif isinstance(e, Mod):
            pass
        elif isinstance(e, FloorDiv):
            if is_one(b):
                return a
        elif isinstance(e, LessThan):
            pass
        elif isinstance(e, Equal):
            pass
        else:
            raise ValueError()

        if isinstance(a, Constant) and isinstance(b, Constant):
            op_dict = {
                Add: operator.add,
                Sub: operator.sub,
                Multiply: operator.mul,
                Div: operator.truediv,
                Mod: operator.mod,
                FloorDiv: operator.floordiv,
                LessThan: operator.lt,
                Equal: operator.eq
            }
            return convert(op_dict[e.__class__](a.value, b.value))
        if a is e.a and b is e.b:
            return e
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

    def visit_Equal(self, e: Equal):
        return self.visit_Binary(e)

    def visit_TensorSlice(self, e: TensorSlice):
        raise NotImplementedError()

    def visit_TensorElement(self, e: TensorElement):
        indices = [self(idx) for idx in e.indices]
        if same(indices, e.indices):
            return e
        else:
            return TensorElement(e.base, indices)

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

    def visit_Call(self, e: Call):
        args = [self(arg) for arg in e.args]
        if same(args, e.args):
            return e
        else:
            return Call(e.func_var, args)

    def visit_Var(self, e: Var):
        return e

    def visit_Axis(self, e: Axis):
        return e

    def visit_Constant(self, e: Constant):
        return e

    def visit_ScalarInput(self, e: ScalarInput):
        return e

    def visit_TensorInput(self, e: TensorInput):
        return e

    def visit_TensorCompute(self, e: TensorCompute):
        shape = [self(v) for v in e.shape]
        axes = [self(v) for v in e.axes]
        value = self(e.value)
        if value is e.value and same(shape, e.shape) and same(axes, e.axes):
            return e
        else:
            return TensorCompute(e.name, shape, axes, value)

    def visit_ReduceCompute(self, e: ReduceCompute):
        value = self(e.value)
        if value is e.value:
            return e
        else:
            return ReduceCompute(value, e.axis, e.reduce_type)


def simplify(expr: Expr):
    simplifier = Simplifier()
    return simplifier(expr)
