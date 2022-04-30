from typing import List, Optional, Dict, Union

from hidet.tos import ops
from hidet.tos.ir.graph import Operator, Tensor
from hidet.tos.ops.definitions.transform import ReshapeOp
from .base import GraphPattern, TensorPattern, OperatorPattern, op_pattern
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

    def source(self) -> TensorPattern:
        return self.z

    def target(self, matched: Dict[Union[TensorPattern, OperatorPattern], Union[Tensor, Operator]]) -> Optional[Tensor]:
        x, scale, y, z = [matched[v] for v in [self.x, self.scale, self.y, self.z]]
        if len(scale.shape) < len(y.shape):
            diff_dims = len(y.shape) - len(scale.shape)
            scale = scale.unsqueeze(dims=list(range(diff_dims)))
        scale_dims = [i for i, dim in enumerate(scale.shape) if dim != 1]
        if len(scale_dims) == 0:
            return ops.reshape(x * ops.flatten(scale), shape=y.shape)
        elif len(scale_dims) == 1:
            dim = reverse_reshape_dim(x.shape, y.shape, scale_dims[0])
            if dim is None:
                return None
            scale = ops.flatten(scale).unsqueeze([i for i in range(len(x.shape)) if i != dim])
            return ops.reshape(x * scale, shape=y.shape)
        else:
            return None


class ReshapeBiasPattern(GraphPattern):
    def __init__(self):
        super().__init__('reshape(x) + bias')
        self.x = TensorPattern.tensor(is_symbolic=True)
        self.bias = TensorPattern.tensor(is_const=True)
        self.y = op_pattern(ReshapeOp, [self.x])
        self.z = self.y + self.bias

    def source(self) -> TensorPattern:
        return self.z

    def target(self, matched: Dict[Union[TensorPattern, OperatorPattern], Union[Tensor, Operator]]) -> Optional[Tensor]:
        x, bias, y, z = [matched[v] for v in [self.x, self.bias, self.y, self.z]]
        if len(bias.shape) < len(y.shape):
            diff_dims = len(y.shape) - len(bias.shape)
            bias = bias.unsqueeze(dims=list(range(diff_dims)))
        scale_dims = [i for i, dim in enumerate(bias.shape) if dim != 1]
        if len(scale_dims) == 0:
            return ops.reshape(x + ops.flatten(bias), shape=y.shape)
        elif len(scale_dims) == 1:
            dim = reverse_reshape_dim(x.shape, y.shape, scale_dims[0])
            if dim is None:
                return None
            bias = ops.flatten(bias).unsqueeze([i for i in range(len(x.shape)) if i != dim])
            return ops.reshape(x + bias, shape=y.shape)
        else:
            return None


def transform_patterns() -> List[GraphPattern]:
    return [
        ReshapeScalePattern(),
        ReshapeBiasPattern()
    ]

