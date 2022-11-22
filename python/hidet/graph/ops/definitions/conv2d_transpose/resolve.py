from typing import List, Optional
from hidet.graph.ir import Tensor
from hidet.graph import ops
from hidet.graph.transforms import ResolveRule, register_resolve_rule

from .conv2d_transpose import Conv2dTransposeOp


@register_resolve_rule(Conv2dTransposeOp)
class Conv2dTransposeResolveRule(ResolveRule):
    def resolve(self, op: Conv2dTransposeOp) -> Optional[List[Tensor]]:
        attrs = op.attrs
        data, weight = op.inputs
        stride = attrs['stride']
        padding = attrs['padding']
        groups = attrs['groups']
        output_padding = attrs['output_padding']
        out = ops.conv2d_transpose_gemm(data, weight, stride, padding, groups, output_padding)
        return [out]
