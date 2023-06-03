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
from typing import List, Optional, Callable, Any

from hidet.ir import dtypes
from hidet.ir.expr import is_constant
from hidet.graph.operator import Operator, Tensor
from hidet.graph.transforms import ResolveRule, register_resolve_rule
from hidet.graph.ops.utils import is_contiguous_norm
from hidet.utils import prod


from .norm import NormalizeOp
from .norm_f16 import normalize_f16


@register_resolve_rule(NormalizeOp)
class NormalizeResolveRule(ResolveRule):
    """
    Resolve a generic reduce operator according to the following rules, in decreasing priority:
    1) resolve_f16: If the input data type is float16, and the size of the last input dimension is an even number,
        return the output of the f16 optimized reduce schedule. (Support of odd number will be added in the future)
    2) resolve_generic: Default case, return the output of the regular f32 reduce schedule.
    """

    def resolve_f16(self, op: Operator) -> Optional[List[Tensor]]:
        dims = op.attrs['dims']
        x: Tensor = op.inputs[0]
        if not is_contiguous_norm(dims, len(x.shape)):
            return None
        if x.dtype != dtypes.float16 or prod([x.shape[dd] for dd in dims]) % 2 != 0:
            return None
        return [normalize_f16(x, dims)]

    def resolve_generic(self, op: Operator) -> Optional[List[Tensor]]:
        dims = op.attrs['dims']
        x: Tensor = op.inputs[0]
        if not is_contiguous_norm(dims, len(x.shape)):
            from hidet.graph.ops import square, rsqrt

            epsilon = op.attrs['epsilon']
            x = x - x.mean(dims, keep_dim=True)
            variance = square(x).mean(dims, keep_dim=True)
            return [x * rsqrt(variance + epsilon)]
        return op.outputs

    def resolve(self, op: Operator) -> Optional[List[Tensor]]:
        assert isinstance(op, NormalizeOp)
        if not is_constant(*op.inputs[0].shape):
            return None
        resolve_funcs: List[Callable[[Operator], Any]] = [self.resolve_f16, self.resolve_generic]
        for resolve_func in resolve_funcs:
            outs = resolve_func(op)
            if outs is not None:
                return outs
        return None
