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

from hidet.ir import dtypes, Expr
from hidet.graph.operator import Operator, Tensor
from hidet.graph.transforms import ResolveRule, register_resolve_rule
from hidet.graph.ops.utils import is_contiguous_dims
from hidet.utils import prod
from .reduce import ReduceBaseOp

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

        if is_contiguous_dims(dims, len(shape)):
            return None
            # for some key models, the reduction dimension spans over multiple dims
            # e.g. 40 x 32 x 32. In this case, it is best to map the reduction as
            # a 2-D tensor so the warp reduction implementation does not need to
            # handle this special case and fusion can still work
            out_shape = op.outputs[0].shape
            spatial = prod(shape[i] for i in range(len(shape)) if i not in dims)
            reduce = prod(shape[i] for i in dims)
            x = x.reshape((spatial, reduce))
            x = op.reforward([x], {"dims": [-1], "keepdims": keepdims})[0]
            x = x.reshape(out_shape)
            return [x]

        if all(shape[d] == 1 for d in dims):
            if keepdims:
                return [x]
            else:
                return [x.squeeze(dims)]

        return None
    def resolve(self, op: Operator) -> Optional[List[Tensor]]:
        assert isinstance(op, ReduceBaseOp)
        resolve_funcs: List[Callable[[Operator], Any]] = [self.resolve_simplify]
        for resolve_func in resolve_funcs:
            outs = resolve_func(op)
            if outs is not None:
                return outs
        return None
