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
# pylint: disable=bad-staticmethod-argument, abstract-method, too-many-ancestors
from .expr_functor import ExprFunctor, ExprRewriter, ExprVisitor
from .compute_functor import ComputeFunctor, ComputeRewriter, ComputeVisitor
from .stmt_functor import StmtFunctor, StmtRewriter, StmtVisitor
from .type_functor import TypeFunctor, TypeRewriter, TypeVisitor
from .mapping_functor import MappingFunctor, MappingRewriter, MappingVisitor
from .layout_functor import LayoutFunctor, LayoutRewriter, LayoutVisitor
from .module_functor import ModuleFunctor, ModuleRewriter, ModuleVisitor


class IRFunctor(ModuleFunctor, StmtFunctor, ComputeFunctor, ExprFunctor, MappingFunctor, LayoutFunctor, TypeFunctor):
    pass


class IRVisitor(ModuleVisitor, StmtVisitor, ComputeVisitor, ExprVisitor, MappingVisitor, LayoutVisitor, TypeVisitor):
    pass


class IRRewriter(
    ModuleRewriter, StmtRewriter, ComputeRewriter, ExprRewriter, MappingRewriter, LayoutRewriter, TypeRewriter
):
    def rewrite(self, node):
        return self.visit(node)
