from typing import List
from hidet.ir.type import ScalarType, TensorType, FuncType
from hidet.ir.expr import BinaryOp, Add, Sub, Multiply, Div, Mod, FloorDiv, Condition, LessThan, Equal, IfThenElse, TensorSlice, Not, Or, And, LessEqual, Let, RightShift, LeftShift, BitwiseNot, BitwiseOr, BitwiseAnd, Neg
from hidet.ir.expr import Var, Constant, TensorElement, Call, Cast
from hidet.ir.dialects.compute import TensorNode, ScalarNode
from hidet.ir.dialects.lowlevel import PointerType, Dereference, Reference, Address, TensorPointerType

from .base import ExprFunctor
from ..dialects.pattern import AnyExpr


def is_bool(tp):
    return isinstance(tp, ScalarType) and tp.name == 'bool'


class TypeInfer(ExprFunctor):
    def visit_Address(self, e: Address):
        base_type = self(e.expr)
        return PointerType(base_type=base_type)

    def visit_Reference(self, e: Reference):
        return self(e.expr)

    def visit_Binary(self, e: BinaryOp):
        a_dtype: ScalarType = self.visit(e.a)
        b_dtype: ScalarType = self.visit(e.b)
        # if not atype or not btype:
        #     return ScalarType(name=None)
        if isinstance(e, (Add, Sub, Multiply, Div, Mod, FloorDiv)):
            return ScalarType(max(a_dtype, b_dtype))
        elif isinstance(e, Condition):
            return ScalarType('bool')
        else:
            raise NotImplementedError('Binary op type infer {}'.format(type(e)))

    def visit_Neg(self, e: Neg):
        return self(e.a)

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

    def visit_LessEqual(self, e: LessEqual):
        return self.visit_Binary(e)

    def visit_And(self, e: And):
        return self.visit_Binary(e)

    def visit_Or(self, e: Or):
        return self.visit_Binary(e)

    def visit_Not(self, e: Not):
        assert is_bool(self.visit(e.a))
        return ScalarType('bool')

    def visit_BitwiseAnd(self, e: BitwiseAnd):
        return self.visit(e.a)

    def visit_BitwiseOr(self, e: BitwiseOr):
        return self.visit(e.a)

    def visit_BitwiseNot(self, e: BitwiseNot):
        return self.visit(e.base)

    def visit_LeftShift(self, e: LeftShift):
        return self.visit(e.base)

    def visit_RightShift(self, e: RightShift):
        return self.visit(e.base)

    def visit_TensorElement(self, e: TensorElement):
        base_type = self.visit(e.base)
        if isinstance(base_type, TensorType):
            return base_type.scalar_type
        elif isinstance(base_type, PointerType):
            return base_type.base_type
        elif isinstance(base_type, TensorPointerType):
            return base_type.tensor_type.scalar_type
        else:
            raise NotImplementedError()

    def visit_TensorSlice(self, e: TensorSlice):
        raise NotImplementedError()

    def visit_IfThenElse(self, e: IfThenElse):
        cond_type = self.visit(e.cond)
        true_type = self.visit(e.then_expr)
        false_type = self.visit(e.else_expr)
        assert is_bool(cond_type)
        if not (isinstance(true_type, ScalarType) and isinstance(false_type, ScalarType) and true_type.name == false_type.name):
            raise ValueError('If-then-else operand 1 and 2 have different types ({} vs {}): {}'.format(true_type, false_type, e))
        return true_type

    def visit_Let(self, e: Let):
        self.visit(e.value)
        return self.visit(e.body)

    def visit_Call(self, e: Call):
        func_var = e.func_var
        func_type = func_var.type
        if not isinstance(func_type, FuncType):
            raise ValueError('Type infer failed, expect a function var "{}" but got variable with type "{}"'.format(func_var, func_type))
        args_type = [self(arg) for arg in e.args]
        return func_type.ret_type_on(args_type)

    def visit_Cast(self, e: Cast):
        return e.target_type

    def visit_Dereference(self, e: Dereference):
        tp = self.visit(e.expr)
        assert isinstance(tp, PointerType)
        return tp.base_type

    def visit_Var(self, e: Var):
        return e.type

    def visit_Constant(self, e: Constant):
        return e.data_type

    def visit_ScalarNode(self, e: ScalarNode):
        return e.data_type

    def visit_TensorNode(self, e: TensorNode):
        return e.data_type

    def visit_AnyExpr(self, e: AnyExpr):
        raise ValueError('Can not infer type of an AnyExpr.')


def infer_type(expr):
    infer = TypeInfer()
    return infer(expr)
