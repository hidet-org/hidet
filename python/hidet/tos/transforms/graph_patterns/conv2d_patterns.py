from typing import List, Optional, Dict, Union

from hidet.tos import ops
from hidet.tos.ir.graph import Operator, Tensor
from hidet.tos.ops.definitions.conv2d_gemm import Conv2dGemmInverseTransformOp
from hidet.tos.ops.definitions.conv2d_winograd import Conv2dWinogradInverseTransformOp
from .base import GraphPattern, TensorPattern, OperatorPattern, op_pattern


class Conv2dGemmInverseScalePattern(GraphPattern):
    def __init__(self):
        super().__init__('conv2d_gemm_inverse(gemm_y) * scale => conv2d_gemm_inverse(gemm_y * scale)')
        self.gemm_y = TensorPattern.tensor(is_symbolic=True)
        self.scale = TensorPattern.tensor()
        self.y = op_pattern(Conv2dGemmInverseTransformOp, [self.gemm_y])
        self.z = self.y * self.scale

    def source(self) -> TensorPattern:
        return self.z

    def target(self, matched: Dict[Union[TensorPattern, OperatorPattern], Union[Tensor, Operator]]) -> Optional[Tensor]:
        gemm_y, y, scale = matched[self.gemm_y], matched[self.y], matched[self.scale]
        if not (scale.shape[0] == scale.shape[2] == scale.shape[3] == 1):
            # we can only fuse the scale on channels
            return None
        return ops.definitions.conv2d_gemm_inverse_transform(gemm_y * scale.squeeze([0, 2, 3]), out_shape=y.shape)


class Conv2dGemmInverseBiasPattern(GraphPattern):
    def __init__(self):
        super().__init__('conv2d_gemm_inverse(gemm_y) + bias => conv2d_gemm_inverse(gemm_y + bias)')
        self.gemm_y = TensorPattern.tensor(is_symbolic=True)
        self.bias = TensorPattern.tensor()
        self.y = op_pattern(Conv2dGemmInverseTransformOp, [self.gemm_y])
        self.z = self.y + self.bias

    def source(self) -> TensorPattern:
        return self.z

    def target(self, matched: Dict[Union[TensorPattern, OperatorPattern], Union[Tensor, Operator]]) -> Optional[Tensor]:
        gemm_y, y, bias = matched[self.gemm_y], matched[self.y], matched[self.bias]
        if not (bias.shape[0] == bias.shape[2] == bias.shape[3] == 1):
            # we can only fuse the scale on channels
            return None
        return ops.definitions.conv2d_gemm_inverse_transform(gemm_y + bias.squeeze([0, 2, 3]), out_shape=y.shape)


class Conv2dWinogradInverseScalePattern(GraphPattern):
    def __init__(self):
        super().__init__('conv2d_winograd_inverse(gemm_y) * scale => conv2d_winograd_inverse(gemm_y * scale)')
        self.gemm_y = TensorPattern.tensor(is_symbolic=True)
        self.scale = TensorPattern.tensor()
        self.y = op_pattern(Conv2dWinogradInverseTransformOp, [self.gemm_y])
        self.z = self.y * self.scale

    def source(self) -> TensorPattern:
        return self.z

    def target(self, matched: Dict[Union[TensorPattern, OperatorPattern], Union[Tensor, Operator]]) -> Optional[Tensor]:
        gemm_y, y, scale = matched[self.gemm_y], matched[self.y], matched[self.scale]
        if not (scale.shape[0] == scale.shape[2] == scale.shape[3] == 1):
            # we can only fuse the scale on channels
            return None
        attrs = y.op.attributes
        return ops.definitions.conv2d_winograd_inverse_transform(
            y=gemm_y * scale.squeeze([0, 3]),
            input_shape=attrs['input_shape'],
            padding=attrs['padding'],
            kernel=attrs['kernel'],
            ms=attrs['ms']
        )


class Conv2dWinogradInverseBiasPattern(GraphPattern):
    def __init__(self):
        super().__init__('conv2d_gemm_inverse(gemm_y) + bias => conv2d_gemm_inverse(gemm_y + bias)')
        self.gemm_y = TensorPattern.tensor(is_symbolic=True)
        self.bias = TensorPattern.tensor()
        self.y = op_pattern(Conv2dWinogradInverseTransformOp, [self.gemm_y])
        self.z = self.y + self.bias

    def source(self) -> TensorPattern:
        return self.z

    def target(self, matched: Dict[Union[TensorPattern, OperatorPattern], Union[Tensor, Operator]]) -> Optional[Tensor]:
        gemm_y, y, bias = matched[self.gemm_y], matched[self.y], matched[self.bias]
        if not (bias.shape[0] == bias.shape[2] == bias.shape[3] == 1):
            # we can only fuse the scale on channels
            return None
        attrs = y.op.attributes
        return ops.definitions.conv2d_winograd_inverse_transform(
            y=gemm_y + bias.squeeze([0, 3]),
            input_shape=attrs['input_shape'],
            padding=attrs['padding'],
            kernel=attrs['kernel'],
            ms=attrs['ms']
        )


def conv2d_patterns() -> List[GraphPattern]:
    return [
        Conv2dGemmInverseScalePattern(),
        Conv2dGemmInverseBiasPattern(),
        Conv2dWinogradInverseScalePattern(),
        Conv2dWinogradInverseBiasPattern()
    ]
