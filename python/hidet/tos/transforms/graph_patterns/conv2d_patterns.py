from typing import List, Optional, Dict, Union

from hidet.tos import ops
from hidet.tos.ir.graph import Tensor
from hidet.tos.ops.definitions.conv2d import Conv2dOp
from .base import GraphPattern, TensorPattern, MatchDict, op_pattern


class Conv2dScalePattern(GraphPattern):
    def __init__(self):
        super().__init__('conv2d(x, w) * scale => conv2d(x, w * scale)')
        self.x = TensorPattern.tensor()
        self.w = TensorPattern.tensor(is_const=True)
        self.scale = TensorPattern.tensor(is_const=True)
        self.y = op_pattern(Conv2dOp, [self.x, self.w])
        self.z = self.y * self.scale

    def source(self) -> List[TensorPattern]:
        return [self.z]

    def target(self, matched: MatchDict) -> Optional[List[Tensor]]:
        x, w, y, scale = [matched[v] for v in [self.x, self.w, self.y, self.scale]]
        if not (scale.shape[0] == scale.shape[2] == scale.shape[3] == 1):
            return None
        attrs = y.op.attrs
        return [ops.conv2d(x, w * scale.squeeze([0]).unsqueeze([3]), stride=attrs['stride'], groups=attrs['groups'])]


def conv2d_patterns() -> List[GraphPattern]:
    return [
        Conv2dScalePattern()
    ]
