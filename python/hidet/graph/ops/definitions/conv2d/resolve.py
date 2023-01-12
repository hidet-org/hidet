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
from hidet.graph.ir import Operator, Tensor
from hidet.graph import ops
from hidet.graph.transforms import ResolveRule, register_resolve_rule

from .conv2d import Conv2dOp


@register_resolve_rule(Conv2dOp)
class Conv2dResolveRule(ResolveRule):
    def __init__(self, enable_winograd=False):
        self.enable_winograd = enable_winograd

    def resolve(self, op: Operator) -> Optional[List[Tensor]]:
        assert isinstance(op, Conv2dOp)
        stride = ops.utils.normalize_stride(op.attrs['stride'])
        groups = op.attrs['groups']
        dilations = op.attrs['dilations']
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
            out = ops.conv2d_gemm(data, weight, stride, dilations, groups)
        return [out]
