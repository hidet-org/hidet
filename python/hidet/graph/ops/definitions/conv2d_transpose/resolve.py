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
