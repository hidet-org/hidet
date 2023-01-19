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
from hidet.graph.ir import Operator, Tensor
from hidet.graph.transforms import ResolveRule, register_resolve_rule

from .reduce import ReduceBaseOp
from .reduce_f16 import reduce_f16


@register_resolve_rule(ReduceBaseOp)
class ReduceResolveRule(ResolveRule):
    """
    Resolve a generic reduce operator according to the following rules, in decreasing priority:
    1) resolve_simplify: If the size of all reduce dimensions are 1, return the input itself if keepdims=True,
        else return Squeeze(input, axis=reduce_dims).
    2) resolve_f16: If the input data type is float16, and the size of the last input dimension is an even number,
        return the output of the f16 optimized reduce schedule. (Support of odd number will be added in the future)
    3) resolve_generic: Default case, return the output of the regular f32 reduce schedule.
    """

    def resolve_simplify(self, op: Operator) -> Optional[List[Tensor]]:
        dims = op.attrs['dims']
        keepdims = op.attrs['keepdims']
        x: Tensor = op.inputs[0]
        shape = x.shape
        if not all(shape[d] == 1 for d in dims):
            return None
        if keepdims:
            return [x]
        return [x.squeeze(dims)]

    def resolve_f16(self, op: Operator) -> Optional[List[Tensor]]:
        dims = op.attrs['dims']
        keepdims = op.attrs['keepdims']
        reduce_type = op.task.attributes['reduce_type']
        x: Tensor = op.inputs[0]
        last_dim = x.shape[-1]
        if x.dtype != dtypes.float16 or last_dim % 2 != 0:
            return None
        return [reduce_f16(x, dims, keepdims, reduce_type)]

    def resolve_generic(self, op: Operator) -> Optional[List[Tensor]]:
        return op.outputs

    def resolve(self, op: Operator) -> Optional[List[Tensor]]:
        assert isinstance(op, ReduceBaseOp)
        resolve_funcs: List[Callable[[Operator], Any]] = [self.resolve_simplify, self.resolve_f16, self.resolve_generic]
        for resolve_func in resolve_funcs:
            outs = resolve_func(op)
            if outs is not None:
                return outs
        return None
