from typing import List, Type, Optional
from hidet.graph.ir import Operator, Tensor
from hidet.graph import ops
from hidet.graph.transforms import ResolveRule, register_resolve_rule

from .conv2d_transpose import Conv2dTransposeOp


@register_resolve_rule
class Conv2dTransposeResolveRule(ResolveRule):
    def op_cls(self) -> Type[Operator]:
        return Conv2dTransposeOp

    def resolve(self, op: Conv2dTransposeOp) -> Optional[List[Tensor]]:
        attrs = op.attrs
        data, weight = op.inputs
        stride = attrs['stride']
        padding = attrs['padding']
        groups = attrs['groups']
        output_padding = attrs['output_padding']
        out = ops.conv2d_transpose_gemm(data, weight, stride, padding, groups, output_padding)
        return [out]
