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
from hidet.graph import ops
from hidet.ir.dtypes import f16
from hidet.graph.transforms.graph_patterns import MatchDict
from hidet.graph.transforms.graph_patterns import op_pattern, register_rewrite_rule, deregister_rewrite_rule
from hidet.graph.transforms.graph_patterns import TensorPattern, SubgraphRewriteRule
from hidet.utils import same_list
from hidet.graph.ops.definitions.matmul import MatmulOp
from hidet.graph.ops.definitions.arithmetic import AddOp, MultiplyScalarOp, DivideScalarOp
from hidet.graph.ops.definitions.activation import SoftmaxOp
from hidet.graph.ops.definitions.attention import attention


class ReorderMulScaleRewriteRule(SubgraphRewriteRule):
    def __init__(self):
        super().__init__(name="matmul(q,k) * c1 => matmul(c1 * q, k)")
        self.q = TensorPattern()
        self.k = TensorPattern()
        self.qk = op_pattern(MatmulOp, [self.q, self.k])
        self.prod = op_pattern(MultiplyScalarOp, [self.qk])

    def source(self):
        return [self.prod]

    def target(self, matched: MatchDict):
        q, k, prod = [matched[t] for t in [self.q, self.k, self.prod]]
        c1 = prod.op.attrs['scalar']
        qc = MultiplyScalarOp(q, c1).get_output(0)
        return [ops.matmul(qc, k)]


class ReorderDivScaleRewriteRule(SubgraphRewriteRule):
    def __init__(self):
        super().__init__(name="matmul(q,k) / c1 => matmul(q / c1, k)")
        self.q = TensorPattern()
        self.k = TensorPattern()
        self.qk = op_pattern(MatmulOp, [self.q, self.k])
        self.div = op_pattern(DivideScalarOp, [self.qk])

    def source(self):
        return [self.div]

    def target(self, matched: MatchDict):
        q, k, div = [matched[t] for t in [self.q, self.k, self.div]]
        c1 = div.op.attrs['scalar']
        qc = DivideScalarOp(q, c1).get_output(0)
        return [ops.matmul(qc, k)]


class AttentionRewriteRule(SubgraphRewriteRule):
    def __init__(self):
        super().__init__(name="matmul(Softmax(matmul(q, k)), v) => attn(q, k, v)")
        self.q = TensorPattern()
        self.k = TensorPattern()
        self.v = TensorPattern()
        self.qk = op_pattern(MatmulOp, [self.q, self.k])
        self.sm = op_pattern(SoftmaxOp, [self.qk])
        self.qkv = op_pattern(MatmulOp, [self.sm, self.v])

    def source(self):
        return [self.qkv]

    def target(self, matched: MatchDict):
        q, k, v = [matched[t] for t in [self.q, self.k, self.v]]
        if (
            q.dtype == k.dtype == v.dtype == f16
            and same_list(q.shape, v.shape)
            and len(q.shape) == len(k.shape)
            and (q.shape[-2], q.shape[-1]) == (k.shape[-1], k.shape[-2])
            and q.shape[-1] <= 160
        ):
            return [attention(q, k, v)]
        else:
            return None


class AttentionMaskAddRewriteRule(SubgraphRewriteRule):
    def __init__(self):
        super().__init__(name="matmul(Softmax(matmul(q, k) + mask), v) => attn(q, k, v, mask)")
        self.q = TensorPattern()
        self.k = TensorPattern()
        self.v = TensorPattern()
        self.mask = TensorPattern()
        self.qk = op_pattern(MatmulOp, [self.q, self.k])
        self.qk_masked = op_pattern(AddOp, [self.qk, self.mask])
        self.sm = op_pattern(SoftmaxOp, [self.qk_masked])
        self.qkv = op_pattern(MatmulOp, [self.sm, self.v])

    def source(self):
        return [self.qkv]

    def target(self, matched: MatchDict):
        q, k, v, mask = [matched[t] for t in [self.q, self.k, self.v, self.mask]]
        if (
            q.dtype == k.dtype == v.dtype == f16
            and same_list(q.shape, v.shape)
            and len(q.shape) == len(k.shape)
            and (q.shape[-2], q.shape[-1]) == (k.shape[-1], k.shape[-2])
            and q.shape[-1] <= 160
        ):
            return [attention(q, k, v, mask)]
        else:
            return None


registered_attn_rules = []


def attn_patterns():
    registered_attn_rules.append(AttentionRewriteRule())
    registered_attn_rules.append(AttentionMaskAddRewriteRule())
    registered_attn_rules.append(ReorderMulScaleRewriteRule())
    registered_attn_rules.append(ReorderDivScaleRewriteRule())
    for attn_rule in registered_attn_rules:
        register_rewrite_rule(attn_rule)


def register_attn_patterns():
    if len(registered_attn_rules) != 0:
        return
    registered_attn_rules.append(AttentionRewriteRule())
    registered_attn_rules.append(AttentionMaskAddRewriteRule())
    registered_attn_rules.append(ReorderMulScaleRewriteRule())
    registered_attn_rules.append(ReorderDivScaleRewriteRule())
    for attn_rule in registered_attn_rules:
        register_rewrite_rule(attn_rule)


def deregister_attn_patterns():
    for attn_rule in registered_attn_rules:
        deregister_rewrite_rule(attn_rule)
    registered_attn_rules.clear()
