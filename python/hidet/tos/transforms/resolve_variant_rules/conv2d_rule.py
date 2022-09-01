from typing import List, Type, Optional
from hidet.tos.ir import Operator, Tensor
from hidet.tos import ops
from hidet.tos.ops.definitions import Conv2dOp
from .base import ResolveRule


class Conv2dResolveRule(ResolveRule):
    def __init__(self, enable_winograd=False):
        self.enable_winograd = enable_winograd

    def op_cls(self) -> Type[Operator]:
        return Conv2dOp

    def resolve(self, op: Operator) -> Optional[List[Tensor]]:
        assert isinstance(op, Conv2dOp)
        stride = ops.utils.normalize_stride(op.attrs['stride'])
        groups = op.attrs['groups']
        channels = op.inputs[1].shape[0]
        if groups == channels:
            return None  # use depthwise schedule in the default Task
        data, weight = op.inputs
        kernel_size = weight.shape[2:]
        if self.enable_winograd and tuple(stride) == (1, 1) and tuple(kernel_size) == (3, 3) and groups == 1:
            # winograd algorithm
            out = ops.conv2d_winograd(data, weight)
        else:
            # implicit gemm algorithm
            out = ops.conv2d_gemm(data, weight, stride, groups)
        return [out]
