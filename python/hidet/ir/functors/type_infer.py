from hidet.ir.type import *
from hidet.ir.expr import *
from hidet.core.compute import *

from .base import ExprFunctor


class TypeInfer(ExprFunctor):
    def visit_Binary(self, e: BinaryOp):
        atype: ScalarType = self.visit(e.a)
        btype: ScalarType = self.visit(e.b)
        assert atype.name == btype.name
        if isinstance(e, (Add, Sub, Multiply, Div, Mod, FloorDiv)):
            return atype
        elif isinstance(e, Condition):
            return ScalarType('bool')

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
        ttype: TensorType = self.visit(e.base)
        shape = []
        for start, end in zip(e.starts, e.ends):
            shape.append(end - start)
        # todo: make the strides correct
        rt = TensorType(ttype.scope, ttype.scalar_type, shape, None)
        return rt

    def visit_TensorElement(self, e: TensorElement):
        return self.visit(e.base).scalar_type

    def visit_Call(self, e: Call):
        # todo
        pass

    def visit_Cast(self, e: Cast):
        return e.target_type

    def visit_Dereference(self, e: Dereference):
        tp = self.visit(e.expr)
        assert isinstance(tp, PointerType)
        return tp.base_type

    def visit_Var(self, e: Var):
        return e.type

    def visit_Axis(self, e: Axis):
        return e.type

    def visit_Constant(self, e: Constant):
        return e.dtype

    def visit_ScalarInput(self, e: ScalarInput):
        return e.dtype

    def visit_TensorInput(self, e: TensorInput):
        return TensorType(None, e.dtype, e.shape, None)

    def visit_TensorCompute(self, e: TensorCompute):
        dtype = self.visit(e.value)
        return TensorType(None, dtype, e.shape, None)

    def visit_ReduceCompute(self, e: ReduceCompute):
        return self.visit(e.value)


def infer_type(expr):
    inferer = TypeInfer()
    return inferer.visit(expr)
