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
from hidet.graph.flow_graph import Tensor, Operator
from hidet.graph.ops.conv2d import Conv2dOp
from hidet.graph.ops.matmul import MatmulOp
from hidet.utils import same_list, initialize
from ..base import SubgraphRewriteRule, TensorPattern, MatchDict, op_pattern, add_rewrite_rule

# we use the heuristic that if one of the inputs to matmul is a constant, and the number if dimensions is two
#   then its a linear layer, and we quantize it

class SymmetricLinearQuantizePatternR(SubgraphRewriteRule):
    def __init__(self, quant_type: str = 'int8', dims=[-1]):
        super().__init__(f'linear(x, w) => linear(x, dequant(wq, scale, {quant_type}, dims={dims}))')
        self.x = TensorPattern.tensor()
        self.w = TensorPattern.tensor(is_const=True)
        self.out = op_pattern(MatmulOp, [self.x, self.w])
        self.quant_type = quant_type
        self.dims = dims
    
    def source(self) -> List[TensorPattern]:
        return [self.out]

    def target(self, matched: MatchDict) -> Optional[List[Tensor]]:
        x, w, out = [matched[v] for v in [self.x, self.w, self.out]]
        if len(w.shape) != 2:
            return None
        attrs = out.op.attrs
        wq, scale = ops.symmetric_quantize(w, quant_type=self.quant_type, dims=self.dims)
        return [ops.matmul(x, ops.symmetric_dequantize(ops.barrier(wq), scale, dims=self.dims), require_prologue=attrs['require_prologue'])]


class SymmetricLinearQuantizePatternL(SubgraphRewriteRule):
    def __init__(self, quant_type: str = 'int8', dims=[-1]):
        super().__init__(f'matmul(w, x) => matmul(dequant(wq, scale, {quant_type}, dims={dims}), x)')
        self.x = TensorPattern.tensor()
        self.w = TensorPattern.tensor(is_const=True)
        self.out = op_pattern(MatmulOp, [self.w, self.x])
        self.quant_type = quant_type
        self.dims = dims
    
    def source(self) -> List[TensorPattern]:
        return [self.out]

    def target(self, matched: MatchDict) -> Optional[List[Tensor]]:
        x, w, out = [matched[v] for v in [self.x, self.w, self.out]]
        attrs = out.op.attrs
        if len(w.shape) != 2:
            return None
        wq, scale = ops.symmetric_quantize(w, quant_type=self.quant_type, dims=self.dims)
        return [ops.matmul(ops.symmetric_dequantize(ops.barrier(wq), scale, dims=self.dims), x, require_prologue=attrs['require_prologue'])]


def symmetric_linear_quantize_patterns(rules: List[SubgraphRewriteRule], quant_type: str = 'int8', dims=[-1]):
    add_rewrite_rule(rules, SymmetricLinearQuantizePatternR(quant_type=quant_type, dims=dims))
    add_rewrite_rule(rules, SymmetricLinearQuantizePatternL(quant_type=quant_type, dims=dims))
    return rules