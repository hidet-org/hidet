from hidet.ir.dialects.lowlevel import Reference, Address, ReferenceType, TensorPointerType, Dereference, Cast, VoidType, PointerType
from hidet.ir.node import Node
from hidet.ir.expr import Call, TensorElement, Not, Or, And, Constant, Var, Let, Equal, LessThan, FloorDiv, Mod, Div, Multiply, Sub, Add, TensorType, ScalarType, Expr, IfThenElse, RightShift, LeftShift, BitwiseNot, BitwiseOr, BitwiseAnd
from hidet.ir.functors import ExprFunctor, TypeFunctor, NodeFunctor
from hidet.ir.type import TypeNode
from hidet.ir.utils.hash_sum import HashSum


class ExprHash(ExprFunctor, TypeFunctor):
    def __init__(self):
        super().__init__()

    def visit(self, e):
        if e in self.memo:
            return self.memo[e]
        if isinstance(e, (str, float, int)):
            ret = HashSum(e)
        elif isinstance(e, tuple):
            ret = HashSum(tuple(self(v) for v in e))
        elif isinstance(e, (Expr, TypeNode)):
            ret = NodeFunctor.visit(self, e)
        else:
            # for stmt/func/...
            ret = HashSum(e)
        self.memo[e] = ret
        return ret

    def hash(self, expr):
        self.memo.clear()
        return self(expr)

    def visit_Var(self, e: Var):
        return HashSum(e) + e.class_index()

    def visit_Constant(self, e: Constant):
        return HashSum(e.value) + self(e.dtype) + e.class_index()

    def visit_Add(self, e: Add):
        return (self(e.a) & self(e.b)) + e.class_index()

    def visit_Sub(self, e: Sub):
        return self(e.a) + self(e.b) + e.class_index()

    def visit_Multiply(self, e: Multiply):
        return (self(e.a) & self(e.b)) + e.class_index()

    def visit_Div(self, e: Div):
        return self(e.a) + self(e.b) + e.class_index()

    def visit_Mod(self, e: Mod):
        return self(e.a) + self(e.b) + e.class_index()

    def visit_FloorDiv(self, e: FloorDiv):
        return self(e.a) + self(e.b) + e.class_index()

    def visit_LessThan(self, e: LessThan):
        return self(e.a) + self(e.b) + e.class_index()

    def visit_LessEqual(self, e: LessThan):
        return self(e.a) + self(e.b) + e.class_index()

    def visit_Equal(self, e: Equal):
        return (self(e.a) & self(e.b)) + e.class_index()

    def visit_IfThenElse(self, e: IfThenElse):
        return self(e.cond) + self(e.then_expr) + self(e.else_expr) + e.class_index()

    def visit_And(self, e: And):
        return (self(e.a) & self(e.b)) + e.class_index()

    def visit_Or(self, e: Or):
        return (self(e.a) & self(e.b)) + e.class_index()

    def visit_Not(self, e: Not):
        return self(e.a) + e.class_index()

    def visit_BitwiseAnd(self, e: BitwiseAnd):
        return (self(e.a) & self(e.b)) + e.class_index()

    def visit_BitwiseOr(self, e: BitwiseOr):
        return (self(e.a) & self(e.b)) + e.class_index()

    def visit_BitwiseNot(self, e: BitwiseNot):
        return self(e.base) + e.class_index()

    def visit_LeftShift(self, e: LeftShift):
        return (self(e.base) + self(e.cnt)) + e.class_index()

    def visit_RightShift(self, e: RightShift):
        return (self(e.base) + self(e.cnt)) + e.class_index()

    def visit_TensorElement(self, e: TensorElement):
        return self(e.base) + self(e.indices) + e.class_index()

    def visit_Cast(self, e: Cast):
        return self(e.expr) + self(e.target_type) + e.class_index()

    def visit_Dereference(self, e: Dereference):
        return self(e.expr) + e.class_index()

    def visit_Address(self, e: Address):
        return self(e.expr) + e.class_index()

    def visit_Reference(self, e: Reference):
        return self(e.expr) + e.class_index()

    def visit_Call(self, e: Call):
        return self(e.func_var) + self(e.args) + e.class_index()

    def visit_Let(self, e: Let):
        return self(e.var) + self(e.value) + self(e.body) + e.class_index()

    def visit_ScalarType(self, t: ScalarType):
        return self(t.name) + t.class_index()

    def visit_TensorType(self, t: TensorType):
        return self(t.scalar_type) + self(t.scope.name) + self(t.shape) + t.class_index()

    def visit_PointerType(self, t: PointerType):
        return self(t.base_type) + t.class_index()

    def visit_TensorPointerType(self, t: TensorPointerType):
        return self(t.tensor_type) + t.class_index()

    def visit_ReferenceType(self, t: ReferenceType):
        return self(t.base_type) + t.class_index()

    def visit_VoidType(self, t: VoidType):
        return t.class_index()

