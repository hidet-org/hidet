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
from hidet.graph.operator import Operator, Tensor
from hidet.graph.transforms import ResolveRule, register_resolve_rule
from hidet.graph import ops
from hidet.ir.expr import is_constant
from hidet.ir.dtypes import float16

from .conv2d import Conv2dOp, Conv2dChannelLastOp
from .conv2d_gemm import parallel_part_heuristic


@register_resolve_rule(Conv2dOp)
class Conv2dResolveRule(ResolveRule):
    def __init__(self, enable_winograd=False):
        self.enable_winograd = enable_winograd

    def resolve(self, op: Operator) -> Optional[List[Tensor]]:
        assert isinstance(op, Conv2dOp)
        padding = op.attrs['padding']
        stride = ops.utils.normalize_stride(op.attrs['stride'])
        groups = op.attrs['groups']
        dilations = op.attrs['dilations']
        channels = op.inputs[1].shape[0]
        # TODO: current assert mechanism does not cover this use case
        if is_constant(channels) and groups == channels:
            return None  # use depthwise schedule in the default Task
        data, weight = op.inputs
        kernel_size = weight.shape[2:]
        if data.dtype == float16 and weight.dtype == float16:
            # we set parallel_k to 1 for channel first, because we need to transpose back;
            #   setting parallel_k > 1 prevents epilogue fusion, leading to bad performance.
            k_parts = 1
            out = ops.conv2d_gemm_fp16(data, weight, padding, stride, dilations, groups, k_parts)
        elif self.enable_winograd and tuple(stride) == (1, 1) and tuple(kernel_size) == (3, 3) and groups == 1:
            # winograd algorithm
            data = ops.conv_pad(data, padding)
            out = ops.conv2d_winograd(data, weight)
        else:
            # implicit gemm algorithm
            data = ops.conv_pad(data, padding)
            out = ops.conv2d_gemm(data, weight, stride, dilations, groups)
        return [out]


@register_resolve_rule(Conv2dChannelLastOp)
class Conv2dChannelLastResolveRule(ResolveRule):
    def resolve(self, op: Operator) -> Optional[List[Tensor]]:
        assert isinstance(op, Conv2dChannelLastOp)
        stride = ops.utils.normalize_stride(op.attrs['stride'])
        groups = op.attrs['groups']
        dilations = op.attrs['dilations']
        padding = op.attrs['padding']
        channels = op.inputs[0].shape[-1]
        # TODO: current assert mechanism does not cover this use case
        if is_constant(channels) and groups == channels:
            return None  # use depthwise schedule in the default Task
        data, weight = op.inputs
        if data.dtype == float16 and weight.dtype == float16:
            # after some benchmarking, basically k_parts = 1 is sufficent for most cases
            if all(is_constant(s) for s in data.shape):
                k_parts = parallel_part_heuristic(data.shape, weight.shape, stride, dilations, groups)
            else:
                k_parts = 1
            out = ops.conv2d_gemm_fp16_channel_last(
                data, weight, padding=padding, stride=stride, dilations=dilations, groups=groups, parallel_k_parts=k_parts
            )
            return [out]
        return None
