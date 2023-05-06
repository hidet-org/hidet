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
from typing import Union, Tuple, Dict

from hidet.ir import Node
from hidet.ir.dialects.pattern import PlaceholderExpr
from hidet.ir.expr import Var, Constant, Add, Sub, Multiply, Div, Mod, FloorDiv, Neg, LessThan, LessEqual
from hidet.ir.expr import NotEqual, Equal, IfThenElse, LogicalAnd, LogicalOr, LogicalNot, BitwiseAnd, BitwiseOr
from hidet.ir.expr import BitwiseNot, BitwiseXor, LeftShift, RightShift
from hidet.ir.expr import TensorSlice, TensorElement, Cast, Dereference, Address, Reference, Call, Let
from hidet.ir.type import ReferenceType, TensorPointerType, VoidType, PointerType, DataType, TensorType
from hidet.ir.utils.hash_sum import HashSum
from hidet.ir.functors import ExprFunctor, TypeFunctor, BaseFunctor


class ExprHash(ExprFunctor, TypeFunctor, BaseFunctor):
    def hash(self, expr):
        self.memo.clear()
        return self(expr)

    def visit_Dict(self, d: Dict):
        return HashSum(tuple(self(k) + self(v) for k, v in d.items()))

    def visit_NotDispatchedNode(self, n: Node):
        raise RuntimeError(f"Node {n} is not supported for hashing.")

    def visit_List(self, e: list):
        return HashSum(tuple(self(v) for v in e))

    def visit_PyConstant(self, c: Union[str, int, float, None]):
        return HashSum(c)

    def visit_Tuple(self, tp: Tuple):
        return HashSum(tuple(self(v) for v in tp))

    def visit_Var(self, e: Var):
        return HashSum(e) + hash(Var)

    def visit_Constant(self, e: Constant):
        return HashSum(e.value) + self(e.type) + hash(Constant)

    def visit_Add(self, e: Add):
        return (self(e.a) & self(e.b)) + hash(Add)

    def visit_Sub(self, e: Sub):
        return self(e.a) + self(e.b) + hash(Sub)

    def visit_Multiply(self, e: Multiply):
        return (self(e.a) & self(e.b)) + hash(Multiply)

    def visit_Div(self, e: Div):
        return self(e.a) + self(e.b) + hash(Div)

    def visit_Mod(self, e: Mod):
        return self(e.a) + self(e.b) + hash(Mod)

    def visit_FloorDiv(self, e: FloorDiv):
        return self(e.a) + self(e.b) + hash(FloorDiv)

    def visit_Neg(self, e: Neg):
        return self(e.a) + hash(Neg)

    def visit_LessThan(self, e: LessThan):
        return self(e.a) + self(e.b) + hash(LessThan)

    def visit_LessEqual(self, e: LessEqual):
        return self(e.a) + self(e.b) + hash(LessEqual)

    def visit_NotEqual(self, e: NotEqual):
        return self(e.a) + self(e.b) + hash(NotEqual)

    def visit_Equal(self, e: Equal):
        return (self(e.a) & self(e.b)) + hash(Equal)

    def visit_IfThenElse(self, e: IfThenElse):
        return self(e.cond) + self(e.then_expr) + self(e.else_expr) + hash(IfThenElse)

    def visit_And(self, e: LogicalAnd):
        return (self(e.a) & self(e.b)) + hash(LogicalAnd)

    def visit_Or(self, e: LogicalOr):
        return (self(e.a) & self(e.b)) + hash(LogicalOr)

    def visit_Not(self, e: LogicalNot):
        return self(e.a) + hash(LogicalNot)

    def visit_BitwiseAnd(self, e: BitwiseAnd):
        return (self(e.a) & self(e.b)) + hash(BitwiseAnd)

    def visit_BitwiseOr(self, e: BitwiseOr):
        return (self(e.a) & self(e.b)) + hash(BitwiseOr)

    def visit_BitwiseNot(self, e: BitwiseNot):
        return self(e.a) + hash(BitwiseNot)

    def visit_BitwiseXor(self, e: BitwiseXor):
        return (self(e.a) & self(e.b)) + hash(BitwiseXor)

    def visit_LeftShift(self, e: LeftShift):
        return (self(e.a) + self(e.b)) + hash(LeftShift)

    def visit_RightShift(self, e: RightShift):
        return (self(e.a) + self(e.b)) + hash(RightShift)

    def visit_TensorElement(self, e: TensorElement):
        return self(e.base) + self(e.indices) + hash(TensorElement)

    def visit_Cast(self, e: Cast):
        return self(e.expr) + self(e.target_type) + hash(Cast)

    def visit_Dereference(self, e: Dereference):
        return self(e.expr) + hash(Dereference)

    def visit_Address(self, e: Address):
        return self(e.expr) + hash(Address)

    def visit_Reference(self, e: Reference):
        return self(e.expr) + hash(Reference)

    def visit_Call(self, e: Call):
        return self(e.func_var) + self(e.args) + hash(Call)

    def visit_Let(self, e: Let):
        return self(e.var) + self(e.value) + self(e.body) + hash(Let)

    def visit_ScalarType(self, t: DataType):
        return self(t.name) + hash(DataType)

    def visit_TensorType(self, t: TensorType):
        return self(t.dtype) + self(t.shape) + hash(TensorType)

    def visit_PointerType(self, t: PointerType):
        return self(t.base_type) + hash(PointerType)

    def visit_TensorPointerType(self, t: TensorPointerType):
        return self(t.tensor_type) + hash(TensorPointerType)

    def visit_ReferenceType(self, t: ReferenceType):
        return self(t.base_type) + hash(ReferenceType)

    def visit_VoidType(self, t: VoidType):
        return hash(VoidType)

    def visit_TensorSlice(self, e: TensorSlice):
        return self(e.base) + self(e.indices) + self(e.starts) + self(e.ends) + hash(TensorSlice)

    def visit_AnyExpr(self, e: PlaceholderExpr):
        return HashSum(e) + hash(PlaceholderExpr)
