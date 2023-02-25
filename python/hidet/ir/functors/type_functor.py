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
from hidet.ir.type import DataType, TensorType, PointerType, TensorPointerType, ReferenceType, VoidType
from hidet.utils import same_list
from .base_functor import BaseFunctor, BaseVisitor, BaseRewriter


class TypeFunctor(BaseFunctor):
    def visit_dispatch(self, node):
        if isinstance(node, DataType):
            return self.visit_ScalarType(node)
        elif isinstance(node, TensorType):
            return self.visit_TensorType(node)
        elif isinstance(node, PointerType):
            return self.visit_PointerType(node)
        elif isinstance(node, TensorPointerType):
            return self.visit_TensorPointerType(node)
        elif isinstance(node, ReferenceType):
            return self.visit_ReferenceType(node)
        elif isinstance(node, VoidType):
            return self.visit_VoidType(node)
        else:
            return NotImplemented

    def visit_ScalarType(self, t: DataType):
        raise NotImplementedError()

    def visit_TensorType(self, t: TensorType):
        raise NotImplementedError()

    def visit_PointerType(self, t: PointerType):
        raise NotImplementedError()

    def visit_TensorPointerType(self, t: TensorPointerType):
        raise NotImplementedError()

    def visit_ReferenceType(self, t: ReferenceType):
        raise NotImplementedError()

    def visit_VoidType(self, t: VoidType):
        raise NotImplementedError()


class TypeVisitor(TypeFunctor, BaseVisitor):
    def visit_ScalarType(self, t: DataType):
        pass

    def visit_TensorType(self, t: TensorType):
        self.visit(t.dtype)
        self.visit(t.shape)
        self.visit(t.layout)

    def visit_PointerType(self, t: PointerType):
        self.visit(t.base_type)

    def visit_TensorPointerType(self, t: TensorPointerType):
        self.visit(t.tensor_type)

    def visit_ReferenceType(self, t: ReferenceType):
        self.visit(t.base_type)

    def visit_VoidType(self, t: VoidType):
        pass


class TypeRewriter(TypeFunctor, BaseRewriter):
    def visit_ScalarType(self, t: DataType):
        return t

    def visit_TensorType(self, t: TensorType):
        dtype = self.visit(t.dtype)
        shape = self.visit(t.shape)
        layout = self.visit(t.layout)
        if same_list([dtype, layout], [t.dtype, t.layout]) and same_list(shape, t.shape):
            return t
        else:
            return TensorType(dtype, shape, layout)

    def visit_PointerType(self, t: PointerType):
        base_type = self.visit(t.base_type)
        if base_type is t.base_type:
            return t
        else:
            return PointerType(base_type)

    def visit_TensorPointerType(self, t: TensorPointerType):
        tensor_type = self.visit(t.tensor_type)
        if tensor_type is t.tensor_type:
            return t
        else:
            return TensorPointerType(tensor_type)

    def visit_ReferenceType(self, t: ReferenceType):
        base_type = self.visit(t.base_type)
        if base_type is t.base_type:
            return t
        else:
            return ReferenceType(base_type)

    def visit_VoidType(self, t: VoidType):
        return t
