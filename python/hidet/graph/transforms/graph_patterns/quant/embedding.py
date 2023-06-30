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

from hidet.graph import ops
from hidet.graph.flow_graph import Tensor
from hidet.graph.ops.transform import TakeOp
from ..base import SubgraphRewriteRule, TensorPattern, MatchDict, op_pattern, add_rewrite_rule

# we use the heuristic that if the weight is constant and the axis is 0 in the take op, then its an embedding layer
#   and we quantize the weights
class SymmetricEmbeddingQuantizePattern(SubgraphRewriteRule):
    def __init__(self, quant_type: str = 'int8'):
        super().__init__(f'embedding(w, ind) => embedding(dequant(wq, scale, {quant_type}), ind)')
        self.w = TensorPattern.tensor(is_const=True)
        self.ind = TensorPattern.tensor()
        self.out = op_pattern(TakeOp, [self.w, self.ind])
        self.quant_type = quant_type

    def source(self) -> List[TensorPattern]:
        return [self.out]

    def target(self, matched: MatchDict) -> Optional[List[Tensor]]:
        w, ind, out = [matched[v] for v in [self.w, self.ind, self.out]]
        attrs = out.op.attrs
        if len(w.shape) != 2 or (not ind.dtype in ('int32', 'int64')) or attrs['axis'] != 0:
            return None
        wq, scale = ops.symmetric_quantize(w, quant_type=self.quant_type, dims=[-1])
        return [ops.take(ops.symmetric_dequantize(ops.barrier(wq), scale, dims=[-1]), ind, axis=0)]


def symmetric_embedding_quantize_patterns(rules: List[SubgraphRewriteRule], quant_type: str = 'int8'):
    add_rewrite_rule(rules, SymmetricEmbeddingQuantizePattern(quant_type=quant_type))
    return rules
