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

from hidet.ir.functors import BaseFunctor, BaseVisitor, BaseRewriter

from hidet.ir.expr import Expr
from hidet.ir.cute import TensorLayout, ComposedTensorLayout, TiledTensorLayout
from hidet.ir.cute.type import TiledTensorType
from hidet.ir.cute.expr import CallOp, Op
from hidet.ir.cute.ops import TiledTensorView, PartitionSrc, PartitionDst, Copy, Mask, Rearrange, Arithmetic
from hidet.ir.cute.collective import CollectiveStore


class CuteFunctor(BaseFunctor):
    def visit_dispatch(self, node):
        if isinstance(node, TiledTensorType):
            return self.visit_TiledTensorType(node)
        elif isinstance(node, CallOp):
            return self.visit_CallOp(node)
        elif isinstance(node, TiledTensorView):
            return self.visit_TiledTensorView(node)
        elif isinstance(node, PartitionSrc):
            return self.visit_PartitionSrc(node)
        elif isinstance(node, PartitionDst):
            return self.visit_PartitionDst(node)
        elif isinstance(node, Copy):
            return self.visit_Copy(node)
        elif isinstance(node, Mask):
            return self.visit_Mask(node)
        elif isinstance(node, Rearrange):
            return self.visit_Rearrange(node)
        elif isinstance(node, CollectiveStore):
            return self.visit_CollectiveStore(node)
        elif isinstance(node, Arithmetic):
            return self.visit_Arithmetic(node)
        elif isinstance(node, Op):
            raise NotImplementedError("Rewriter for the following op is not implemented: \n{}".format(node.op_name()))
        else:
            return NotImplemented

    def visit_CallOp(self, call: CallOp):
        raise NotImplementedError()

    def visit_TiledTensorType(self, t: TiledTensorType):
        raise NotImplementedError()

    def visit_TiledTensorView(self, e: TiledTensorView):
        raise NotImplementedError()

    def visit_PartitionSrc(self, e: PartitionSrc):
        raise NotImplementedError()

    def visit_PartitionDst(self, e: PartitionDst):
        raise NotImplementedError()

    def visit_Mask(self, e: Mask):
        raise NotImplementedError()

    def visit_Copy(self, e: Copy):
        raise NotImplementedError()

    def visit_Rearrange(self, e: Rearrange):
        raise NotImplementedError()

    def visit_CollectiveStore(self, e: CollectiveStore):
        raise NotImplementedError()

    def visit_Arithmetic(self, e: Arithmetic):
        raise NotImplementedError()

    def visit_Layout(self, layout):
        if isinstance(layout, TensorLayout):
            return self.visit_TensorLayout(layout)
        elif isinstance(layout, TiledTensorLayout):
            return self.visit_TiledTensorLayout(layout)
        elif isinstance(layout, ComposedTensorLayout):
            return self.visit_ComposedTensorLayout(layout)
        elif isinstance(layout, Op):
            raise NotImplementedError(f"Visitor for the following layout is not implemented: \n{layout}")
        else:
            return NotImplemented

    def visit_TensorLayout(self, n: TensorLayout):
        raise NotImplementedError()

    def visit_ComposedTensorLayout(self, n: ComposedTensorLayout):
        raise NotImplementedError()

    def visit_TiledTensorLayout(self, n: TiledTensorLayout):
        raise NotImplementedError()


class CuteVisitor(CuteFunctor, BaseVisitor):
    def visit_TiledTensorType(self, t: TiledTensorType):
        self.visit(t.dtype)
        self.visit_Layout(t.layout)

    def visit_CallOp(self, call: CallOp):
        self.visit(call.op)

    def visit_TiledTensorView(self, e: TiledTensorView):
        self.visit(e.args)

    def visit_PartitionSrc(self, e: PartitionSrc):
        self.visit(e.args)

    def visit_PartitionDst(self, e: PartitionDst):
        self.visit(e.args)

    def visit_Mask(self, e: Mask):
        self.visit(e.args)

    def visit_Copy(self, e: Copy):
        self.visit(e.args)

    def visit_Rearrange(self, e: Rearrange):
        self.visit(e.args)
        self.visit_Layout(e.layout)

    def visit_CollectiveStore(self, e: CollectiveStore):
        self.visit(e.args)

    def visit_Arithmetic(self, e: Arithmetic):
        self.visit(e.args)

    def visit_TensorLayout(self, n: TensorLayout):
        pass

    def visit_TiledTensorLayout(self, n: TiledTensorLayout):
        pass

    def visit_ComposedTensorLayout(self, n: ComposedTensorLayout):
        self.visit(n.base)


class CuteRewriter(CuteFunctor, BaseRewriter):
    def visit_TiledTensorType(self, t: TiledTensorType):
        tp = self.visit(t.dtype)
        ly = self.visit_Layout(t.layout)
        if tp is t.dtype and ly is t.layout:
            return t
        else:
            return TiledTensorType(tp, layout=ly, scope=t.scope)

    def visit_CallOp(self, call: CallOp):
        op = self.visit(call.op)
        if op is call.op:
            return call
        else:
            if isinstance(op, Op):
                return op.make_call()
            else:
                assert isinstance(op, Expr)
                return op

    def visit_TiledTensorView(self, e: TiledTensorView):
        x = self.visit(e.x)
        layout = self.visit_Layout(e.layout)
        if x is e.x and layout is e.layout:
            return e
        else:
            return e.reforward([x], attrs_update={"layout": layout})

    def visit_PartitionSrc(self, e: PartitionSrc):
        x = self.visit(e.x)
        if x is e.x:
            return e
        else:
            return e.reforward([x])

    def visit_PartitionDst(self, e: PartitionDst):
        x = self.visit(e.x)
        if x is e.x:
            return e
        else:
            return e.reforward([x])

    def visit_Mask(self, e: Mask):
        extents = [self.visit(v) for v in e.extents]
        if all(x is y for x, y in zip(extents, e.extents)):
            return e
        else:
            return e.reforward(extents)

    def visit_Copy(self, e: Copy):
        src = self.visit(e.src)
        dst = self.visit(e.dst)
        if e.mask is not None:
            mask = self.visit(e.mask)
        else:
            mask = None
        if src is e.src and dst is e.dst and mask is e.mask:
            return e
        else:
            return e.reforward([src, dst, mask])

    def visit_Rearrange(self, e: Rearrange):
        x = self.visit(e.x)
        layout = self.visit_Layout(e.layout)
        if x is e.x and layout is e.layout:
            return e
        else:
            return e.reforward([x], attrs_update={"layout": layout})

    def visit_CollectiveStore(self, e: CollectiveStore):
        src = self.visit(e.src)
        dst = self.visit(e.dst)
        offsets = [self.visit(e) for e in e.offsets]
        args = [src, dst] + offsets
        if e.extents is not None:
            extents = [self.visit(extent) for extent in e.extents]
        else:
            extents = None
        if src is e.src and dst is e.dst and (extents is None or all(e1 is e2 for e1, e2 in zip(extents, e.extents))):
            return e
        else:
            return e.reforward(args + extents if extents else args)

    def visit_Arithmetic(self, e: Arithmetic):
        args = [self.visit(arg) for arg in e.args]
        if all(x is y for x, y in zip(args, e.args)):
            return e
        else:
            return e.reforward(args)

    def visit_TensorLayout(self, n: TensorLayout):
        return n

    def visit_TiledTensorLayout(self, n: TiledTensorLayout):
        return n

    def visit_ComposedTensorLayout(self, n: ComposedTensorLayout):
        layout = self.visit_Layout(n.layout)
        base = self.visit(n.base)
        if layout is n.layout and base is n.base:
            return n
        else:
            return ComposedTensorLayout(layout, base, n.functor)
