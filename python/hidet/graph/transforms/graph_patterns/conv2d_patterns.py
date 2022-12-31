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
from hidet.graph.ir.flow_graph import Tensor, Operator
from hidet.graph.ops.definitions.conv2d import Conv2dOp
from hidet.utils import same_list, initialize
from .base import SubgraphRewriteRule, TensorPattern, MatchDict, op_pattern, register_rewrite_rule


class Conv2dScalePattern(SubgraphRewriteRule):
    def __init__(self):
        super().__init__('conv2d(x, w) * scale => conv2d(x, w * scale)')
        self.x = TensorPattern.tensor()
        self.w = TensorPattern.tensor(is_const=True)
        self.scale = TensorPattern.tensor(is_const=True)
        self.y = op_pattern(Conv2dOp, [self.x, self.w])
        self.z = self.y * self.scale

    def source(self) -> List[TensorPattern]:
        return [self.z]

    def target(self, matched: MatchDict) -> Optional[List[Tensor]]:
        x, w, y, scale = [matched[v] for v in [self.x, self.w, self.y, self.scale]]
        if not scale.shape[0] == scale.shape[2] == scale.shape[3] == 1:
            return None
        attrs = y.op.attrs
        return [ops.conv2d(x, w * scale.squeeze([0]).unsqueeze([3]), stride=attrs['stride'], groups=attrs['groups'])]


class TwoConv2dFusionPattern(SubgraphRewriteRule):
    def __init__(self):
        super().__init__('conv2d(x, w1)|conv2d(x, w2) => conv2d(x, w1 + w2)')
        self.x = TensorPattern.tensor()
        self.w1 = TensorPattern.tensor(is_const=True)
        self.w2 = TensorPattern.tensor(is_const=True)
        self.y1 = op_pattern(Conv2dOp, [self.x, self.w1])
        self.y2 = op_pattern(Conv2dOp, [self.x, self.w2])

    def source(self) -> List[TensorPattern]:
        return [self.y1, self.y2]

    def target(self, matched: MatchDict) -> Optional[List[Tensor]]:
        x, w1, w2, y1, y2 = [matched[v] for v in [self.x, self.w1, self.w2, self.y1, self.y2]]
        op1: Operator = y1.op
        op2: Operator = y2.op
        if op1.attrs['groups'] == op2.attrs['groups'] == 1:
            if same_list(op1.attrs['stride'], op2.attrs['stride']):
                if same_list(w1.shape[1:], w2.shape[1:]):
                    w = ops.concat([w1, w2], axis=0)
                    y = ops.conv2d(x, w, stride=op1.attrs['stride'], groups=1)
                    # pylint: disable=unbalanced-tuple-unpacking
                    new_y1, new_y2 = ops.split(y, axis=1, parts=[w1.shape[0], w2.shape[0]])
                    return [new_y1, new_y2]
        return None


class ThreeConv2dFusionPattern(SubgraphRewriteRule):
    def __init__(self):
        super().__init__('conv2d(x, w1)|conv2d(x, w2)|conv2d(x, w3) => conv2d(x, w1 + w2 + w3)')
        self.x = TensorPattern.tensor()
        self.w1 = TensorPattern.tensor(is_const=True)
        self.w2 = TensorPattern.tensor(is_const=True)
        self.w3 = TensorPattern.tensor(is_const=True)
        self.y1 = op_pattern(Conv2dOp, [self.x, self.w1])
        self.y2 = op_pattern(Conv2dOp, [self.x, self.w2])
        self.y3 = op_pattern(Conv2dOp, [self.x, self.w3])

    def source(self) -> List[TensorPattern]:
        return [self.y1, self.y2, self.y3]

    def target(self, matched: MatchDict) -> Optional[List[Tensor]]:
        x, w1, w2, w3, y1, y2, y3 = [matched[v] for v in [self.x, self.w1, self.w2, self.w3, self.y1, self.y2, self.y3]]
        op1: Operator = y1.op
        op2: Operator = y2.op
        op3: Operator = y3.op
        if op1.attrs['groups'] == op2.attrs['groups'] == op3.attrs['groups'] == 1:
            if same_list(op1.attrs['stride'], op2.attrs['stride']):
                if same_list(op1.attrs['stride'], op3.attrs['stride']):
                    if same_list(w1.shape[1:], w2.shape[1:]) and same_list(w1.shape[1:], w3.shape[1:]):
                        w = ops.concat([w1, w2, w3], axis=0)
                        y = ops.conv2d(x, w, stride=op1.attrs['stride'], groups=1)
                        # pylint: disable=unbalanced-tuple-unpacking
                        new_y1, new_y2, new_y3 = ops.split(y, axis=1, parts=[w1.shape[0], w2.shape[0], w3.shape[0]])
                        return [new_y1, new_y2, new_y3]
        return None


@initialize()
def conv2d_patterns():
    register_rewrite_rule(Conv2dScalePattern())
    register_rewrite_rule(ThreeConv2dFusionPattern())  # put this before TwoConv2dFusionPattern
    register_rewrite_rule(TwoConv2dFusionPattern())
