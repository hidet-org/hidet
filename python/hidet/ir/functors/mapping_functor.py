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
from hidet.ir.mapping import SpatialTaskMapping, RepeatTaskMapping, ComposedTaskMapping
from .base_functor import BaseFunctor, BaseVisitor, BaseRewriter


class MappingFunctor(BaseFunctor):
    def visit_dispatch(self, node):
        if isinstance(node, SpatialTaskMapping):
            return self.visit_SpatialTaskMapping(node)
        elif isinstance(node, RepeatTaskMapping):
            return self.visit_RepeatTaskMapping(node)
        elif isinstance(node, ComposedTaskMapping):
            return self.visit_ComposedTaskMapping(node)
        else:
            return NotImplemented

    def visit_SpatialTaskMapping(self, mapping: SpatialTaskMapping):
        raise NotImplementedError()

    def visit_RepeatTaskMapping(self, mapping: RepeatTaskMapping):
        raise NotImplementedError()

    def visit_ComposedTaskMapping(self, mapping: ComposedTaskMapping):
        raise NotImplementedError()


class MappingVisitor(BaseVisitor, MappingFunctor):
    def visit_SpatialTaskMapping(self, mapping: SpatialTaskMapping):
        pass

    def visit_RepeatTaskMapping(self, mapping: RepeatTaskMapping):
        pass

    def visit_ComposedTaskMapping(self, mapping: ComposedTaskMapping):
        self.visit(mapping.outer)
        self.visit(mapping.inner)


class MappingRewriter(BaseRewriter, MappingFunctor):
    def visit_SpatialTaskMapping(self, mapping: SpatialTaskMapping):
        return mapping

    def visit_RepeatTaskMapping(self, mapping: RepeatTaskMapping):
        return mapping

    def visit_ComposedTaskMapping(self, mapping: ComposedTaskMapping):
        outer = self.visit(mapping.outer)
        inner = self.visit(mapping.inner)
        if outer is mapping.outer and inner is mapping.inner:
            return mapping
        else:
            return ComposedTaskMapping(outer, inner)
