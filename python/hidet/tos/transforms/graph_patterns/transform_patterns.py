from typing import List, Optional

from hidet.tos import ops
from hidet.tos.ir.graph import Tensor
from hidet.tos.ops.definitions.transform import ReshapeOp, SqueezeOp, StridedSliceOp
from .base import GraphPattern, TensorPattern, MatchDict, op_pattern
from hidet.utils import prod


def reverse_reshape_dim(orig_shape, new_shape, new_axis) -> Optional[int]:
    pre_sum = prod(new_shape[:new_axis])
    cnt = 1
    for i, extent in enumerate(orig_shape):
        if cnt == pre_sum:
            if len(orig_shape) == i:
                return None
            elif orig_shape[i] == new_shape[new_axis]:
                return i
            else:
                return None
        elif cnt > pre_sum:
            return None
        else:
            cnt *= extent
    return None


class ReshapeScalePattern(GraphPattern):
    def __init__(self):
        super().__init__('reshape(x) * scale')
        self.x = TensorPattern.tensor(is_symbolic=True)
        self.scale = TensorPattern.tensor(is_const=True)
        self.y = op_pattern(ReshapeOp, [self.x])
        self.z = self.y * self.scale

    def source(self) -> List[TensorPattern]:
        return [self.z]

    def target(self, matched: MatchDict) -> Optional[List[Tensor]]:
        x, scale, y, z = [matched[v] for v in [self.x, self.scale, self.y, self.z]]
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


class ReshapeBiasPattern(GraphPattern):
    def __init__(self):
        super().__init__('reshape(x) + bias')
        self.x = TensorPattern.tensor(is_symbolic=True)
        self.bias = TensorPattern.tensor(is_const=True)
        self.y = op_pattern(ReshapeOp, [self.x])
        self.z = self.y + self.bias

    def source(self) -> List[TensorPattern]:
        return [self.z]

    def target(self, matched: MatchDict) -> Optional[List[Tensor]]:
        x, bias, y, z = [matched[v] for v in [self.x, self.bias, self.y, self.z]]
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


class SqueezeMultiplyPattern(GraphPattern):
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
        c = c.unsqueeze(dims)   # now, c has the same shape as x
        return [ops.squeeze(x * c, dims=dims)]


def transform_patterns() -> List[GraphPattern]:
    return [
        ReshapeScalePattern(),
        ReshapeBiasPattern(),
        SqueezeMultiplyPattern()
    ]

