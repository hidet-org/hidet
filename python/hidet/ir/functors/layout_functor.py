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
from hidet.ir.layout import DataLayout, StridesLayout, LocalLayout, ComposedLayout, SwizzleLayout, ConcatLayout
from hidet.utils import same_list
from .base_functor import BaseFunctor, BaseVisitor, BaseRewriter


class LayoutFunctor(BaseFunctor):
    def visit_dispatch(self, node):
        if isinstance(node, DataLayout):
            if isinstance(node, StridesLayout):
                return self.visit_StridesLayout(node)
            elif isinstance(node, LocalLayout):
                return self.visit_LocalLayout(node)
            elif isinstance(node, ComposedLayout):
                return self.visit_ComposedLayout(node)
            elif isinstance(node, SwizzleLayout):
                return self.visit_SwizzleLayout(node)
            elif isinstance(node, ConcatLayout):
                return self.visit_ConcatLayout(node)
            else:
                raise ValueError('Can not recognize layout {}'.format(node))
        else:
            return NotImplemented

    def visit_StridesLayout(self, layout: StridesLayout):
        raise NotImplementedError()

    def visit_LocalLayout(self, layout: LocalLayout):
        raise NotImplementedError()

    def visit_ComposedLayout(self, layout: ComposedLayout):
        raise NotImplementedError()

    def visit_SwizzleLayout(self, layout: SwizzleLayout):
        raise NotImplementedError()

    def visit_ConcatLayout(self, layout: ConcatLayout):
        raise NotImplementedError()


class LayoutVisitor(BaseVisitor, LayoutFunctor):
    def visit_StridesLayout(self, layout: StridesLayout):
        self.visit(layout.size)
        self.visit(layout.shape)
        self.visit(layout.strides)

    def visit_LocalLayout(self, layout: LocalLayout):
        self.visit(layout.shape)

    def visit_ComposedLayout(self, layout: ComposedLayout):
        self.visit(layout.outer)
        self.visit(layout.inner)

    def visit_SwizzleLayout(self, layout: SwizzleLayout):
        self.visit(layout.base)
        self.visit(layout.shape)
        self.visit(layout.size)

    def visit_ConcatLayout(self, layout: ConcatLayout):
        self.visit(layout.lhs)
        self.visit(layout.rhs)


class LayoutRewriter(BaseRewriter, LayoutFunctor):
    def visit_StridesLayout(self, layout: StridesLayout):
        size = self.visit(layout.size)
        shape = self.visit(layout.shape)
        strides = self.visit(layout.strides)
        if same_list([size], [layout.size]) and same_list(shape, layout.shape) and same_list(strides, layout.strides):
            return layout
        else:
            return StridesLayout(shape, strides)

    def visit_LocalLayout(self, layout: LocalLayout):
        shape = self.visit(layout.shape)
        if same_list(shape, layout.shape):
            return layout
        else:
            return LocalLayout(shape)

    def visit_ComposedLayout(self, layout: ComposedLayout):
        outer = self.visit(layout.outer)
        inner = self.visit(layout.inner)
        if outer is layout.outer and inner is layout.inner:
            return layout
        else:
            return ComposedLayout(outer, inner)

    def visit_SwizzleLayout(self, layout: SwizzleLayout):
        base = self.visit(layout.base)
        if base is layout.base:
            return layout
        else:
            return SwizzleLayout(base, layout.dim, layout.regards_dim, layout.log_step)

    def visit_ConcatLayout(self, layout: ConcatLayout):
        lhs = self.visit(layout.lhs)
        rhs = self.visit(layout.rhs)
        if lhs is layout.lhs and rhs is layout.rhs:
            return layout
        else:
            return ConcatLayout(lhs, rhs)
