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

from hidet.utils import prod
from hidet.ir.expr import is_constant
from hidet.graph.operator import Operator, Tensor
from hidet.graph.transforms import ResolveRule, register_resolve_rule
from hidet.graph.ops.utils import is_contiguous_dims
from .norm import NormalizeOp, normalize


@register_resolve_rule(NormalizeOp)
class NormalizeResolveRule(ResolveRule):
    """
    Resolve a generic reduce operator according to the following rules, in decreasing priority:
    1) resolve_f16: If the input data type is float16, and the size of the last input dimension is an even number,
        return the output of the f16 optimized reduce schedule. (Support of odd number will be added in the future)
    2) resolve_generic: Default case, return the output of the regular f32 reduce schedule.
    """

    def resolve_generic(self, op: Operator) -> Optional[List[Tensor]]:
        dims = op.attrs['dims']
        x: Tensor = op.inputs[0]

        if not is_contiguous_dims(dims, len(x.shape)):
            from hidet.graph.ops import square, rsqrt

            epsilon = op.attrs['epsilon']
            x = x - x.mean(dims, keep_dim=True)
            variance = square(x).mean(dims, keep_dim=True)
            return [x * rsqrt(variance + epsilon)]
        elif len(dims) > 1:
            shape = x.shape
            spatial = prod(shape[i] for i in range(len(shape)) if i not in dims)
            reduce = prod(shape[i] for i in dims)
            x = x.reshape((spatial, reduce))
            x = normalize(x, [-1], op.attrs['epsilon'], op.attrs['accumulate_dtype'])
            x = x.reshape(shape)
        return None

    def resolve(self, op: Operator) -> Optional[List[Tensor]]:
        assert isinstance(op, NormalizeOp)
        if not is_constant(*op.inputs[0].shape):
            return None
        resolve_funcs: List[Callable[[Operator], Any]] = [self.resolve_generic]
        for resolve_func in resolve_funcs:
            outs = resolve_func(op)
            if outs is not None:
                return outs
        return None
