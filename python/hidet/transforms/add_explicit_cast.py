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
from hidet.ir.functors import StmtExprRewriter, TypeInfer
from hidet.ir.stmt import Stmt, AssignStmt, BufferStoreStmt
from hidet.ir.expr import Expr, Cast, Add, Sub, Multiply, Div, BinaryOp, cast
from hidet.ir.type import DataType, TypeNode, TensorType, TensorPointerType, PointerType, ReferenceType, VoidType
from .base import FunctionBodyPass, Pass


class TypeNotMatch(Exception):
    def __init__(self, a, b, msg=""):
        super().__init__()
        self.a = a
        self.b = b
        self.msg = msg


class TypeChecker:
    def visit(self, a: TypeNode, b: TypeNode):
        if isinstance(a, DataType):
            return self.visit_ScalarType(a, b)
        elif isinstance(a, TensorType):
            return self.visit_TensorType(a, b)
        elif isinstance(a, PointerType):
            return self.visit_PointerType(a, b)
        elif isinstance(a, TensorPointerType):
            return self.visit_TensorPointerType(a, b)
        elif isinstance(a, ReferenceType):
            return self.visit_ReferenceType(a, b)
        elif isinstance(a, VoidType):
            return self.visit_VoidType(a, b)
        else:
            raise ValueError('Can not recognize type {}'.format(a))

    @staticmethod
    def check(a, b, cond, msg=""):
        if not cond:
            raise TypeNotMatch(a, b, msg)

    def visit_ScalarType(self, a: DataType, b: TypeNode):
        self.check(a, b, isinstance(b, DataType))
        assert isinstance(b, DataType)
        self.check(a, b, a.name == b.name)

    def visit_TensorType(self, a: TensorType, b: TypeNode):
        self.check(a, b, isinstance(b, TensorType))
        assert isinstance(b, TensorType)
        self.visit(a.dtype, b.dtype)
        # todo: check data layout and shape

    def visit_PointerType(self, a: PointerType, b: TypeNode):
        self.check(a, b, isinstance(b, PointerType))
        assert isinstance(b, PointerType)
        self.visit(a.base_type, b.base_type)

    def visit_TensorPointerType(self, a: TensorPointerType, b: TypeNode):
        self.check(a, b, isinstance(b, TensorPointerType))
        assert isinstance(b, TensorPointerType)
        self.visit(a.tensor_type, b.tensor_type)

    def visit_ReferenceType(self, a: ReferenceType, b: TypeNode):
        self.check(a, b, isinstance(b, ReferenceType))
        assert isinstance(b, ReferenceType)
        self.visit(a.base_type, b.base_type)

    def visit_VoidType(self, a: VoidType, b: TypeNode):
        self.check(a, b, isinstance(b, VoidType))


def same_type(a: TypeNode, b: TypeNode) -> bool:
    try:
        TypeChecker().visit(a, b)
        return True
    except TypeNotMatch:
        return False


class AddExplicitCastRewriter(StmtExprRewriter):
    def __init__(self):
        super().__init__()
        self.type_infer = TypeInfer()

    @staticmethod
    def convert(source_type: TypeNode, target_type: TypeNode, source_value: Expr) -> Expr:
        if isinstance(source_type, DataType) and isinstance(target_type, DataType):
            # because there is no implicit conversion function between bfloat16 and float16
            # in the underlying cuda c library, we use 'float32' as a bridge type
            has_float16 = 'float16' in [source_type.name, target_type.name]
            has_bfloat16 = 'bfloat16' in [source_type.name, target_type.name]
            if has_float16 and has_bfloat16:
                return Cast(Cast(source_value, 'float32'), target_type)
        if same_type(source_type, target_type):
            return source_value
        else:
            return Cast(source_value, target_type)

    def visit_Binary(self, e: BinaryOp):
        if isinstance(e, (Add, Sub, Multiply, Div)):
            from hidet.ir.utils.type_utils import numeric_promotion

            a, b = self(e.a), self(e.b)
            a_dtype: DataType = self.type_infer(a)
            b_dtype: DataType = self.type_infer(b)
            if a_dtype.name != b_dtype.name:
                op = e.__class__
                c_dtype = numeric_promotion(a_dtype, b_dtype)
                if a_dtype.name == c_dtype.name:
                    return op(a, cast(b, a_dtype))
                else:
                    return op(cast(a, b_dtype), b)
            else:
                return StmtExprRewriter.visit_Binary(self, e)
        else:
            return StmtExprRewriter.visit_Binary(self, e)

    def visit_Cast(self, e: Cast):
        expr = self(e.expr)
        source_type = self.type_infer(expr)
        target_type = e.target_type
        return self.convert(source_type, target_type, expr)

    def visit_AssignStmt(self, stmt: AssignStmt):
        value = self(stmt.value)
        var = self(stmt.var)
        source_type = self.type_infer(value)
        target_type = self.type_infer(var)
        return AssignStmt(var, self.convert(source_type, target_type, value))

    def visit_BufferStoreStmt(self, stmt: BufferStoreStmt):
        value = self(stmt.value)
        buf = self(stmt.buf)
        indices = self(stmt.indices)
        source_type = self.type_infer(value)
        buffer_type = self.type_infer(buf)
        if isinstance(buffer_type, TensorType):
            target_type = buffer_type.dtype
        elif isinstance(buffer_type, TensorPointerType):
            target_type = buffer_type.tensor_type.dtype
        elif isinstance(buffer_type, PointerType):
            target_type = buffer_type.base_type
        else:
            raise ValueError('Can not recognize the buffer type: {}'.format(buffer_type))
        return BufferStoreStmt(buf, indices, self.convert(source_type, target_type, source_value=value))


class AddExplicitCastPass(FunctionBodyPass):
    def process_body(self, stmt: Stmt) -> Stmt:
        rewriter = AddExplicitCastRewriter()
        return rewriter(stmt)


def add_explicit_cast_pass() -> Pass:
    return AddExplicitCastPass()
