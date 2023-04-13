from hidet.graph import ops
from hidet.ir.dtypes import f16
from hidet.graph.transforms.graph_patterns import MatchDict
from hidet.graph.transforms.graph_patterns import op_pattern, register_rewrite_rule
from hidet.graph.transforms.graph_patterns import TensorPattern, SubgraphRewriteRule
from hidet.utils import same_list
from hidet.graph.ops.definitions.matmul import MatmulOp
from hidet.graph.ops.definitions.arithmetic import AddOp, MultiplyScalarOp, DivideScalarOp
from hidet.graph.ops.definitions.activation import SoftmaxOp
from hidet.graph.ops.definitions.transform import CastOp
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


# This may not be needed anymore, as it was added to Hidet
class RemoveCastRewriteRule(SubgraphRewriteRule):
    def __init__(self):
        super().__init__(name="cast(cast(a, *), a.dtype) => a")
        self.a = TensorPattern()
        self.cast1 = op_pattern(CastOp, [self.a])
        self.cast2 = op_pattern(CastOp, [self.cast1])

    def source(self):
        return [self.cast2]

    def target(self, matched: MatchDict):
        a, cast2 = [matched[t] for t in [self.a, self.cast2]]
        atype = a.dtype
        c2type = cast2.op.attrs['dtype']
        if atype == c2type:
            return [a]
        else:
            return None


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


def attn_patterns():
    register_rewrite_rule(AttentionRewriteRule())
    register_rewrite_rule(AttentionMaskAddRewriteRule())
    register_rewrite_rule(ReorderMulScaleRewriteRule())
    register_rewrite_rule(ReorderDivScaleRewriteRule())
