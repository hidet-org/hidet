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
from typing import List, Optional, Union
from hidet.graph.ir import Tensor
from hidet.graph import ops
from hidet.graph.transforms import ResolveRule, register_resolve_rule

from .arithmetic import PowOp


@register_resolve_rule(PowOp)
class PowOpResolveRule(ResolveRule):
    def resolve(self, op: PowOp) -> Optional[List[Tensor]]:
        a, b = op.inputs
        assert isinstance(b, Tensor)
        if b.is_symbolic():
            return None
        if len(b.shape) != 0:
            return None
        b: Union[float, int, bool] = b.item()
        if b in [2.0, 2]:
            return [ops.square(a)]
        elif b in [1.0, 1]:
            return [a]
        elif b in [-1.0, -1]:
            return [ops.reciprocal(a)]
        else:
            return None
