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
from hidet.ir.type import TensorType, tensor_type, PointerType, TensorPointerType
from hidet.ir.expr import Var, TensorElement, TensorSlice, Constant
from hidet.ir.stmt import BufferStoreStmt, DeclareStmt
from hidet.ir.func import Function
from hidet.ir.functors import simplify_to_int, FuncStmtExprRewriter
from hidet.transforms import Pass
from hidet.ir.layout import StridesLayout, DataLayout


class FlattenTensorAccessRewriter(FuncStmtExprRewriter):
    # flatten all high-dimension tensor access
    # A = int[3, 4]
    #   TensorElement:  A[2, 1]     ==> A[2 * 4 + 1]
    # BufferStoreStmt:  A[2, 1] = 3 ==> A[2 * 4 + 1] = 3
    def visit_Function(self, func: Function):
        for var in func.params:
            if isinstance(var.type, TensorType):
                size = simplify_to_int(var.type.layout.size)
                self.memo[var] = Var(var.hint, tensor_type(var.type.dtype, [size], DataLayout.row_major([size])))
            elif isinstance(var.type, TensorPointerType):
                self.memo[var] = var
        body = self(func.body)
        params = [self(p) for p in func.params]
        return Function(
            func.name, params, body, func.ret_type, kind=func.kind, extern_vars=func.extern_vars, attrs=func.attrs
        )

    @staticmethod
    def get_layout(e) -> DataLayout:
        if isinstance(e, Var):
            if isinstance(e.type, TensorType):
                return e.type.layout
            elif isinstance(e.type, TensorPointerType):
                return e.type.tensor_type.layout
            elif isinstance(e.type, PointerType):
                return StridesLayout(shape=[0], strides=[1])
        elif isinstance(e, Constant) and isinstance(e.type, TensorType):
            return e.type.layout
        raise ValueError("Can not infer layout from '{}' (expression {})".format(type(e), e))

    def visit_DeclareStmt(self, stmt: DeclareStmt):
        if isinstance(stmt.var.type, TensorType):
            size = simplify_to_int(stmt.var.type.layout.size)
            var = Var(stmt.var.hint, tensor_type(stmt.var.type.dtype, [size], DataLayout.row_major([size])))
            self.memo[stmt.var] = var
            init = self(stmt.init) if stmt.init is not None else None
            return DeclareStmt(var, init, scope=stmt.scope)
        else:
            return FuncStmtExprRewriter.visit_DeclareStmt(self, stmt)

    def visit_TensorElement(self, e: TensorElement):
        var = self(e.base)
        indices = [self(i) for i in e.indices]
        layout = self.get_layout(e.base)
        if len(indices) != len(layout.shape):
            raise ValueError(
                'Access {}-d tensor {} named {} with {}-d indices {}'.format(
                    len(layout.shape), list(layout.shape), var.hint, len(indices), list(indices)
                )
            )
        global_index = layout(*indices)
        return TensorElement(var, [global_index])

    def visit_BufferStoreStmt(self, stmt: BufferStoreStmt):
        var = self(stmt.buf)
        indices = [self(i) for i in stmt.indices]
        value = self(stmt.value)
        layout = self.get_layout(stmt.buf)
        global_index = layout(indices)
        return BufferStoreStmt(var, [global_index], value)

    def visit_TensorSlice(self, e: TensorSlice):
        raise ValueError('there should not be any tensor slice after flattening tensor slice. got\n{}'.format(e))


class FlattenTensorIndexPass(Pass):
    def process_func(self, func: Function) -> Function:
        flatten_index = FlattenTensorAccessRewriter()
        return flatten_index(func)


def flatten_tensor_index_pass():
    return FlattenTensorIndexPass()
