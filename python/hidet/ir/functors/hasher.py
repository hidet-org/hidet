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
from hidet.ir.compute import TensorNode, ScalarNode
from hidet.ir.dialects.pattern import AnyExpr
from hidet.ir.expr import Expr, Var, Constant, Add, Sub, Multiply, Div, Mod, FloorDiv, Neg, LessThan, LessEqual
from hidet.ir.expr import (
    NotEqual,
    Equal,
    IfThenElse,
    LogicalAnd,
    LogicalOr,
    LogicalNot,
    BitwiseAnd,
    BitwiseOr,
    BitwiseNot,
    BitwiseXor,
)
from hidet.ir.expr import LeftShift, RightShift, TensorElement, Cast, Dereference, Address, Reference, Call, Let
from hidet.ir.expr import TensorSlice
from hidet.ir.functors import ExprFunctor, TypeFunctor, NodeFunctor
from hidet.ir.type import TypeNode, ReferenceType, TensorPointerType, VoidType, PointerType, DataType, TensorType
from hidet.ir.utils.hash_sum import HashSum


class ExprHash(ExprFunctor, TypeFunctor):
    def __init__(self):
        super().__init__()

    def visit(self, node):
        if node in self.memo:
            return self.memo[node]
        if isinstance(node, (str, float, int)):
            ret = HashSum(node)
        elif isinstance(node, tuple):
            ret = HashSum(tuple(self(v) for v in node))
        elif isinstance(node, (Expr, TypeNode)):
            ret = NodeFunctor.visit(self, node)
        elif node is None:
            ret = HashSum(None)
        else:
            # for stmt/func/...
            ret = HashSum(node)
        self.memo[node] = ret
        return ret

    def hash(self, expr):
        self.memo.clear()
        return self(expr)

    def visit_Var(self, e: Var):
        return HashSum(e) + e.class_index()

    def visit_Constant(self, e: Constant):
        return HashSum(e.value) + self(e.type) + e.class_index()

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

    def visit_Neg(self, e: Neg):
        return self(e.a) + e.class_index()

    def visit_LessThan(self, e: LessThan):
        return self(e.a) + self(e.b) + e.class_index()

    def visit_LessEqual(self, e: LessEqual):
        return self(e.a) + self(e.b) + e.class_index()

    def visit_NotEqual(self, e: NotEqual):
        return self(e.a) + self(e.b) + e.class_index()

    def visit_Equal(self, e: Equal):
        return (self(e.a) & self(e.b)) + e.class_index()

    def visit_IfThenElse(self, e: IfThenElse):
        return self(e.cond) + self(e.then_expr) + self(e.else_expr) + e.class_index()

    def visit_And(self, e: LogicalAnd):
        return (self(e.a) & self(e.b)) + e.class_index()

    def visit_Or(self, e: LogicalOr):
        return (self(e.a) & self(e.b)) + e.class_index()

    def visit_Not(self, e: LogicalNot):
        return self(e.a) + e.class_index()

    def visit_BitwiseAnd(self, e: BitwiseAnd):
        return (self(e.a) & self(e.b)) + e.class_index()

    def visit_BitwiseOr(self, e: BitwiseOr):
        return (self(e.a) & self(e.b)) + e.class_index()

    def visit_BitwiseNot(self, e: BitwiseNot):
        return self(e.base) + e.class_index()

    def visit_BitwiseXor(self, e: BitwiseXor):
        return (self(e.a) & self(e.b)) + e.class_index()

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

    def visit_ScalarType(self, t: DataType):
        return self(t.name) + t.class_index()

    def visit_TensorType(self, t: TensorType):
        return self(t.dtype) + self(t.shape) + t.class_index()

    def visit_PointerType(self, t: PointerType):
        return self(t.base_type) + t.class_index()

    def visit_TensorPointerType(self, t: TensorPointerType):
        return self(t.tensor_type) + t.class_index()

    def visit_ReferenceType(self, t: ReferenceType):
        return self(t.base_type) + t.class_index()

    def visit_VoidType(self, t: VoidType):
        return t.class_index()

    def visit_TensorSlice(self, e: TensorSlice):
        return self(e.base) + self(e.indices) + self(e.starts) + self(e.ends) + e.class_index()

    def visit_ScalarNode(self, e: ScalarNode):
        raise NotImplementedError()

    def visit_TensorNode(self, e: TensorNode):
        raise NotImplementedError()

    def visit_AnyExpr(self, e: AnyExpr):
        return HashSum(e) + e.class_index()
