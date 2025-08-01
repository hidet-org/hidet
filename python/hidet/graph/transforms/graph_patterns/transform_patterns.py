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
from typing import List, Optional

from hidet.graph import ops
from hidet.graph.flow_graph import Tensor
from hidet.graph.ops.transform import ReshapeOp, SqueezeOp, CastOp
from hidet.graph.ops.arithmetic import UnaryElementwiseOp, BinaryElementwiseOp
from hidet.graph.ops.arithmetic import UnaryElementwiseOperation, BinaryElementwiseOperation
from hidet.utils import prod, initialize
from hidet.ir.expr import is_true
from .base import SubgraphRewriteRule, TensorPattern, MatchDict, op_pattern, register_rewrite_rule


def reverse_reshape_dim(orig_shape, new_shape, new_axis) -> Optional[int]:
    pre_sum = prod(new_shape[:new_axis])
    cnt = 1
    for i, extent in enumerate(orig_shape):
        if is_true(cnt == pre_sum):
            if len(orig_shape) == i:
                return None
            elif is_true(orig_shape[i] == new_shape[new_axis]):
                return i
            else:
                return None
        elif is_true(cnt > pre_sum):
            return None
        else:
            cnt *= extent
    return None


class ReshapeScalePattern(SubgraphRewriteRule):
    def __init__(self):
        super().__init__('reshape(x) * scale')
        self.x = TensorPattern.tensor(is_symbolic=True)
        self.scale = TensorPattern.tensor(is_const=True)
        self.y = op_pattern(ReshapeOp, [self.x])
        self.z = self.y * self.scale

    def source(self) -> List[TensorPattern]:
        return [self.z]

    def target(self, matched: MatchDict) -> Optional[List[Tensor]]:
        x, scale, y = [matched[v] for v in [self.x, self.scale, self.y]]
        if len(scale.shape) < len(y.shape):
            diff_dims = len(y.shape) - len(scale.shape)
            scale = scale.unsqueeze(dims=list(range(diff_dims)))
        scale_dims = [i for i, dim in enumerate(scale.shape) if dim != 1]
        if len(scale_dims) == 0:
            return [ops.reshape(x * ops.flatten(scale), shape=y.shape)]
        elif len(scale_dims) == 1:
            dim = reverse_reshape_dim(x.shape, y.shape, scale_dims[0])
            if dim is None:
                return None
            scale = ops.flatten(scale).unsqueeze([i for i in range(len(x.shape)) if i != dim])
            return [ops.reshape(x * scale, shape=y.shape)]
        else:
            return None


class ReshapeBiasPattern(SubgraphRewriteRule):
    def __init__(self):
        super().__init__('reshape(x) + bias')
        self.x = TensorPattern.tensor(is_symbolic=True)
        self.bias = TensorPattern.tensor(is_const=True)
        self.y = op_pattern(ReshapeOp, [self.x])
        self.z = self.y + self.bias

    def source(self) -> List[TensorPattern]:
        return [self.z]

    def target(self, matched: MatchDict) -> Optional[List[Tensor]]:
        x, bias, y = [matched[v] for v in [self.x, self.bias, self.y]]
        if len(bias.shape) < len(y.shape):
            diff_dims = len(y.shape) - len(bias.shape)
            bias = bias.unsqueeze(dims=list(range(diff_dims)))
        scale_dims = [i for i, dim in enumerate(bias.shape) if dim != 1]
        if len(scale_dims) == 0:
            return [ops.reshape(x + ops.flatten(bias), shape=y.shape)]
        elif len(scale_dims) == 1:
            dim = reverse_reshape_dim(x.shape, y.shape, scale_dims[0])
            if dim is None:
                return None
            bias = ops.flatten(bias).unsqueeze([i for i in range(len(x.shape)) if i != dim])
            return [ops.reshape(x + bias, shape=y.shape)]
        else:
            return None


class SqueezeMultiplyPattern(SubgraphRewriteRule):
    def __init__(self):
        super().__init__('squeeze(x) * c => squeeze(x * c)')
        self.x = TensorPattern.tensor()
        self.c = TensorPattern.tensor(is_const=True)
        self.s = op_pattern(SqueezeOp, [self.x])
        self.y = self.s * self.c

    def source(self) -> List[TensorPattern]:
        return [self.y]

    def target(self, matched: MatchDict) -> Optional[List[Tensor]]:
        x, c, s, y = matched[self.x], matched[self.c], matched[self.s], matched[self.y]
        dims = s.op.attrs['dims']
        if len(c.shape) < len(y.shape):
            c = c.unsqueeze(list(range(len(y.shape) - len(c.shape))))
        c = c.unsqueeze(dims)  # now, c has the same shape as x
        return [ops.squeeze(x * c, dims=dims)]


class FanoutTwoCast(SubgraphRewriteRule):
    def __init__(self):
        super().__init__('y1 = cast(x), y2 = cast(x) => y1 = y2 = z = cast(x)')
        self.x = TensorPattern.tensor(is_symbolic=True)
        self.c1 = op_pattern(CastOp, [self.x])
        self.c2 = op_pattern(CastOp, [self.x])

    def source(self) -> List[TensorPattern]:
        return [self.c1, self.c2]

    def target(self, matched: MatchDict) -> Optional[List[Tensor]]:
        x, c1, c2 = [matched[v] for v in [self.x, self.c1, self.c2]]
        if c1.dtype != c2.dtype:
            return None
        z = ops.cast(x, dtype=c1.dtype)
        return [z, z]


class FanoutThreeCast(SubgraphRewriteRule):
    def __init__(self):
        super().__init__('y1,2,3 = cast(x) => y1 = y2 = y3 = z = cast(x)')
        self.x = TensorPattern.tensor(is_symbolic=True)
        self.c1 = op_pattern(CastOp, [self.x])
        self.c2 = op_pattern(CastOp, [self.x])
        self.c3 = op_pattern(CastOp, [self.x])

    def source(self) -> List[TensorPattern]:
        return [self.c1, self.c2, self.c3]

    def target(self, matched: MatchDict) -> Optional[List[Tensor]]:
        x, c1, c2, c3 = [matched[v] for v in [self.x, self.c1, self.c2, self.c3]]
        if not (c1.dtype == c2.dtype and c2.dtype == c3.dtype):
            return None
        z = ops.cast(x, dtype=c1.dtype)
        return [z, z, z]


class DoubleCast(SubgraphRewriteRule):
    def __init__(self):
        super().__init__('cast(cast(x)) => x')
        self.x = TensorPattern.tensor(is_symbolic=True)
        self.c1 = op_pattern(CastOp, [self.x])
        self.c2 = op_pattern(CastOp, [self.c1])

    def source(self) -> List[TensorPattern]:
        return [self.c2]

    def target(self, matched: MatchDict) -> Optional[List[Tensor]]:
        x, _, c2 = [matched[v] for v in [self.x, self.c1, self.c2]]
        if not c2.dtype == x.dtype:
            return None
        return [x]


class MultiInputCompositeElementwiseLeftRightRewriteRule(SubgraphRewriteRule):
    def __init__(self):
        super().__init__('binaryOp(unaryOp(x), unaryOp(y)) => compositeOp(x, y)')
        self.x1 = TensorPattern()
        self.x2 = TensorPattern()
        self.y1 = op_pattern(UnaryElementwiseOp, [self.x1])
        self.y2 = op_pattern(UnaryElementwiseOp, [self.x2])
        self.z = op_pattern(BinaryElementwiseOp, [self.y1, self.y2])

    def source(self) -> List[TensorPattern]:
        return [self.z]

    def target(self, matched: MatchDict) -> Optional[List[Tensor]]:
        x1, x2, y1, y2, z = [matched[v] for v in [self.x1, self.x2, self.y1, self.y2, self.z]]
        left_unary_op: UnaryElementwiseOperation = y1.op.op
        right_unary_op: UnaryElementwiseOperation = y2.op.op
        if left_unary_op.name != right_unary_op.name:
            return None
        binary_op: BinaryElementwiseOperation = z.op.op
        out = ops.arithmetic.composite_multi_input_elementwise(x1, x2, left_unary_op, right_unary_op, binary_op)
        return [out]


class CompositeElementwiseLeftRightRewriteRule(SubgraphRewriteRule):
    def __init__(self):
        super().__init__('binaryOp(unaryOp_left(x), unaryOp_right(x)) => compositeOp(x)')
        self.x = TensorPattern()
        self.y1 = op_pattern(UnaryElementwiseOp, [self.x])
        self.y2 = op_pattern(UnaryElementwiseOp, [self.x])
        self.z = op_pattern(BinaryElementwiseOp, [self.y1, self.y2])

    def source(self) -> List[TensorPattern]:
        return [self.z]

    def target(self, matched: MatchDict) -> Optional[List[Tensor]]:
        x, y1, y2, z = [matched[v] for v in [self.x, self.y1, self.y2, self.z]]
        left_unary_op: UnaryElementwiseOperation = y1.op.op
        right_unary_op: UnaryElementwiseOperation = y2.op.op
        binary_op: BinaryElementwiseOperation = z.op.op
        out = ops.arithmetic.composite_elementwise(x, left_unary_op, right_unary_op, binary_op)
        return [out]


class CompositeElementwiseLeftRewriteRule(SubgraphRewriteRule):
    def __init__(self):
        super().__init__('binaryOp(unaryOp(x), x) => compositeOp(x)')
        self.x = TensorPattern()
        self.y1 = op_pattern(UnaryElementwiseOp, [self.x])
        self.z = op_pattern(BinaryElementwiseOp, [self.y1, self.x])

    def source(self) -> List[TensorPattern]:
        return [self.z]

    def target(self, matched: MatchDict) -> Optional[List[Tensor]]:
        x, y1, z = [matched[v] for v in [self.x, self.y1, self.z]]
        left_unary_op: UnaryElementwiseOperation = y1.op.op
        binary_op: BinaryElementwiseOperation = z.op.op
        out = ops.arithmetic.composite_elementwise(x, left_unary_op, None, binary_op)
        return [out]


class CompositeElementwiseRightRewriteRule(SubgraphRewriteRule):
    def __init__(self):
        super().__init__('binaryOp(x, unaryOp(x)) => compositeOp(x)')
        self.x = TensorPattern()
        self.y2 = op_pattern(UnaryElementwiseOp, [self.x])
        self.z = op_pattern(BinaryElementwiseOp, [self.x, self.y2])

    def source(self) -> List[TensorPattern]:
        return [self.z]

    def target(self, matched: MatchDict) -> Optional[List[Tensor]]:
        x, y2, z = [matched[v] for v in [self.x, self.y2, self.z]]
        right_unary_op: UnaryElementwiseOperation = y2.op.op
        binary_op: BinaryElementwiseOperation = z.op.op
        out = ops.arithmetic.composite_elementwise(x, None, right_unary_op, binary_op)
        return [out]


@initialize()
def transform_patterns():
    register_rewrite_rule(ReshapeScalePattern())
    register_rewrite_rule(ReshapeBiasPattern())
    register_rewrite_rule(SqueezeMultiplyPattern())
    register_rewrite_rule(FanoutTwoCast())
    register_rewrite_rule(FanoutThreeCast())
    register_rewrite_rule(DoubleCast())
    register_rewrite_rule(MultiInputCompositeElementwiseLeftRightRewriteRule())
    register_rewrite_rule(CompositeElementwiseLeftRightRewriteRule())
    register_rewrite_rule(CompositeElementwiseLeftRewriteRule())
    register_rewrite_rule(CompositeElementwiseRightRewriteRule())
