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
from hidet.ir.functors import IRRewriter
from hidet.ir.tools import TypeInfer
from hidet.ir.stmt import AssignStmt, BufferStoreStmt
from hidet.ir.expr import Expr, Cast, Add, Sub, Multiply, Div, BinaryExpr, cast
from hidet.ir.type import DataType, BaseType, TensorType, TensorPointerType, PointerType, ReferenceType, VoidType
from hidet.ir.type import ArrayType, StringType
from hidet.ir.func import Function
from .base import FunctionPass, Pass


class TypeNotMatch(Exception):
    def __init__(self, a, b, msg=""):
        super().__init__()
        self.a = a
        self.b = b
        self.msg = msg


class TypeChecker:
    def visit(self, a: BaseType, b: BaseType):
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
        elif isinstance(a, StringType):
            return self.visit_StringType(a, b)
        elif isinstance(a, ArrayType):
            return self.visit_ArrayType(a, b)
        else:
            raise ValueError('Can not recognize type {}'.format(a))

    @staticmethod
    def check(a, b, cond, msg=""):
        if not cond:
            raise TypeNotMatch(a, b, msg)

    def visit_ScalarType(self, a: DataType, b: BaseType):
        self.check(a, b, isinstance(b, DataType))
        assert isinstance(b, DataType)
        self.check(a, b, a.name == b.name)

    def visit_TensorType(self, a: TensorType, b: BaseType):
        self.check(a, b, isinstance(b, TensorType))
        assert isinstance(b, TensorType)
        self.visit(a.dtype, b.dtype)
        # todo: check data layout and shape

    def visit_PointerType(self, a: PointerType, b: BaseType):
        self.check(a, b, isinstance(b, PointerType))
        assert isinstance(b, PointerType)
        self.visit(a.base_type, b.base_type)

    def visit_TensorPointerType(self, a: TensorPointerType, b: BaseType):
        self.check(a, b, isinstance(b, TensorPointerType))
        assert isinstance(b, TensorPointerType)
        self.visit(a.tensor_type, b.tensor_type)

    def visit_ReferenceType(self, a: ReferenceType, b: BaseType):
        self.check(a, b, isinstance(b, ReferenceType))
        assert isinstance(b, ReferenceType)
        self.visit(a.base_type, b.base_type)

    def visit_VoidType(self, a: VoidType, b: BaseType):
        self.check(a, b, isinstance(b, VoidType))

    def visit_StringType(self, a: StringType, b: BaseType):
        self.check(a, b, isinstance(b, StringType))

    def visit_ArrayType(self, a: ArrayType, b: BaseType):
        self.check(a, b, isinstance(b, (ArrayType, PointerType)))


def same_type(a: BaseType, b: BaseType) -> bool:
    try:
        TypeChecker().visit(a, b)
        return True
    except TypeNotMatch:
        return False


class AddExplicitCastRewriter(IRRewriter):
    def __init__(self):
        super().__init__()
        self.type_infer = TypeInfer()

    @staticmethod
    def convert(source_type: BaseType, target_type: BaseType, source_value: Expr) -> Expr:
        if isinstance(source_type, DataType) and isinstance(target_type, DataType):
            # because there is no implicit conversion function between bfloat16 and float16
            # in the underlying cuda c library, we use 'float32' as a bridge type
            has_float16 = 'float16' in [source_type.name, target_type.name]
            has_bfloat16 = 'bfloat16' in [source_type.name, target_type.name]
            if has_float16 and has_bfloat16:
                return cast(cast(source_value, 'float32'), target_type)
        if same_type(source_type, target_type):
            return source_value
        else:
            return cast(source_value, target_type)

    def visit_Binary(self, e: BinaryExpr):
        a, b = self(e.a), self(e.b)
        a_type: BaseType = self.type_infer(a)
        b_type: BaseType = self.type_infer(b)
        if a_type.is_data_type() and b_type.is_data_type() and isinstance(e, (Add, Sub, Multiply, Div)):
            from hidet.ir.utils.type_utils import numeric_promotion

            a_dtype: DataType = a_type.as_data_type()
            b_dtype: DataType = b_type.as_data_type()
            if a_dtype.name != b_dtype.name:
                op = e.__class__
                c_dtype = numeric_promotion(a_dtype, b_dtype)
                if a_dtype.name == c_dtype.name:
                    return op(a, self.convert(b_dtype, a_dtype, b))
                else:
                    return op(self.convert(a_dtype, b_dtype, a), b)
            else:
                return IRRewriter.visit_Binary(self, e)
        else:
            return IRRewriter.visit_Binary(self, e)

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
        elif isinstance(buffer_type, ArrayType):
            target_type = buffer_type.base_type
        else:
            raise ValueError('Can not recognize the buffer type: {}'.format(buffer_type))
        return BufferStoreStmt(buf, indices, self.convert(source_type, target_type, source_value=value))


class AddExplicitCastPass(FunctionPass):
    def process_func(self, func: Function) -> Function:
        rewriter = AddExplicitCastRewriter()
        return rewriter.rewrite(func)


def add_explicit_cast_pass() -> Pass:
    return AddExplicitCastPass()
