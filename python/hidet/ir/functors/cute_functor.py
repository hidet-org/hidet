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
from hidet.ir.cute.type import TiledTensorType
from hidet.ir.cute.expr import CallOp, Op
from hidet.ir.cute.ops import TiledTensorView, PartitionSrc, PartitionDst


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


class CuteVisitor(CuteFunctor, BaseVisitor):
    def visit_TiledTensorType(self, t: TiledTensorType):
        self.visit(t.dtype)

    def visit_CallOp(self, call: CallOp):
        self.visit(call.op)

    def visit_TiledTensorView(self, e: TiledTensorView):
        self.visit(e.args)

    def visit_PartitionSrc(self, e: PartitionSrc):
        self.visit(e.args)

    def visit_PartitionDst(self, e: PartitionDst):
        self.visit(e.args)


class CuteRewriter(CuteFunctor, BaseRewriter):
    def visit_TiledTensorType(self, t: TiledTensorType):
        tp = self.visit(t.dtype)
        if tp is t.dtype:
            return t
        else:
            return TiledTensorType(tp, layout=t.layout, scope=t.scope)

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
        if x is e.x:
            return e
        else:
            return e.reforward([x])

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
