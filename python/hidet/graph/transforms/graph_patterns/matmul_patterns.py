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

import hidet
from hidet.graph import ops
from hidet.graph.flow_graph import Tensor
from hidet.graph.ops.matmul import MatmulOp
from hidet.graph.ops.transform import TransposeOp2D
from hidet.utils import same_list, initialize
from hidet.ir.expr import is_constant

# pylint: disable=unused-import
from .base import SubgraphRewriteRule, TensorPattern, MatchDict, op_pattern, register_rewrite_rule


class TwoMatmulFusionPattern(SubgraphRewriteRule):
    def __init__(self):
        super().__init__('matmul(x, c1)|matmul(x, c2) ==> matmul(x, concat(c1, c2)) followed by split')
        self.x = TensorPattern.tensor(is_symbolic=True)
        self.c1 = TensorPattern.tensor(is_const=True)
        self.c2 = TensorPattern.tensor(is_const=True)
        self.y1 = op_pattern(MatmulOp, [self.x, self.c1])
        self.y2 = op_pattern(MatmulOp, [self.x, self.c2])

    def source(self) -> List[TensorPattern]:
        return [self.y1, self.y2]

    def target(self, matched: MatchDict) -> Optional[List[Tensor]]:
        x, c1, c2 = [matched[t] for t in [self.x, self.c1, self.c2]]
        y1, y2 = [matched[t] for t in [self.y1, self.y2]]

        # Previously, resolving `nn.Linear` to `matmul_nt` will break the CI test
        # because this fusion does not consider whether the second matrix is transposed.
        tr1, tr2 = [op.attrs['transpose_b'] for op in [y1.op, y2.op]]
        if tr1 != tr2:
            return None

        if not len(c1.shape) == len(c2.shape) >= 2:
            return None

        if not tr1 and same_list(c1.shape[:-1], c2.shape[:-1]):
            c = ops.concat([c1, c2], axis=-1)
            y = ops.matmul(x, c)
            # pylint: disable=unbalanced-tuple-unpacking
            new_y1, new_y2 = ops.split(y, axis=-1, parts_or_sections=[c1.shape[-1], c2.shape[-1]])
            return [new_y1, new_y2]

        elif tr1 and same_list(c1.shape[:-2] + (c1.shape[-1],), c2.shape[:-2] + (c2.shape[-1],)):
            c = ops.concat([c1, c2], axis=-2)
            y = ops.matmul_nt(x, c)
            new_y1, new_y2 = ops.split(y, axis=-1, parts_or_sections=[c1.shape[-2], c2.shape[-2]])
            return [new_y1, new_y2]
        return None


class ThreeMatmulFusionPattern(SubgraphRewriteRule):
    def __init__(self):
        super().__init__('matmul(x, c1)|matmul(x, c2)|matmul(x, c3) => matmul(x, concat(c1, c2, c3)) followed by split')
        self.x = TensorPattern.tensor(is_symbolic=True)
        self.c1 = TensorPattern.tensor(is_const=True)
        self.c2 = TensorPattern.tensor(is_const=True)
        self.c3 = TensorPattern.tensor(is_const=True)
        self.y1 = op_pattern(MatmulOp, [self.x, self.c1])
        self.y2 = op_pattern(MatmulOp, [self.x, self.c2])
        self.y3 = op_pattern(MatmulOp, [self.x, self.c3])

    def source(self) -> List[TensorPattern]:
        return [self.y1, self.y2, self.y3]

    def target(self, matched: MatchDict) -> Optional[List[Tensor]]:
        x, c1, c2, c3 = [matched[t] for t in [self.x, self.c1, self.c2, self.c3]]

        tr1, tr2, tr3 = [op.attrs['transpose_b'] for op in [matched[t].op for t in [self.y1, self.y2, self.y3]]]
        if tr1 != tr2 or tr2 != tr3 or tr1 != tr3:
            return None

        if not len(c1.shape) == len(c2.shape) == len(c3.shape) >= 2:
            return None

        if not tr1 and same_list(c1.shape[:-1], c2.shape[:-1]) and same_list(c2.shape[:-1], c3.shape[:-1]):
            c = ops.concat([c1, c2, c3], axis=-1)
            y = ops.matmul(x, c)
            # pylint: disable=unbalanced-tuple-unpacking
            new_y1, new_y2, new_y3 = ops.split(y, axis=-1, parts_or_sections=[c1.shape[-1], c2.shape[-1], c3.shape[-1]])
            return [new_y1, new_y2, new_y3]

        elif (
            tr1
            and same_list(c1.shape[:-2] + (c1.shape[-1],), c2.shape[:-2] + (c2.shape[-1],))
            and same_list(c2.shape[:-2] + (c2.shape[-1],), c3.shape[:-2] + (c3.shape[-1],))
        ):
            c = ops.concat([c1, c2, c3], axis=-2)
            y = ops.matmul_nt(x, c)
            new_y1, new_y2, new_y3 = ops.split(y, axis=-1, parts_or_sections=[c1.shape[-2], c2.shape[-2], c3.shape[-2]])
            return [new_y1, new_y2, new_y3]

        return None


class ThreeMatmulBiasFusionPattern(SubgraphRewriteRule):
    def __init__(self):
        super().__init__('3 branches of matmul(x, branch c) + branch b ==> matmul(x, c) + b followed by split')
        self.x = TensorPattern.tensor(is_symbolic=True)
        self.c1 = TensorPattern.tensor(is_const=True)
        self.c2 = TensorPattern.tensor(is_const=True)
        self.c3 = TensorPattern.tensor(is_const=True)
        self.b1 = TensorPattern.tensor(is_const=True)
        self.b2 = TensorPattern.tensor(is_const=True)
        self.b3 = TensorPattern.tensor(is_const=True)
        self.y1 = op_pattern(MatmulOp, [self.x, self.c1]) + self.b1
        self.y2 = op_pattern(MatmulOp, [self.x, self.c2]) + self.b2
        self.y3 = op_pattern(MatmulOp, [self.x, self.c3]) + self.b3

    def source(self) -> List[TensorPattern]:
        return [self.y1, self.y2, self.y3]

    def target(self, matched: MatchDict) -> Optional[List[Tensor]]:
        x, c1, c2, c3, b1, b2, b3, y1, y2, y3 = [
            matched[t]
            for t in [self.x, self.c1, self.c2, self.c3, self.b1, self.b2, self.b3, self.y1, self.y2, self.y3]
        ]

        if not len(c1.shape) == len(c2.shape) == len(c3.shape) >= 2:
            return None

        tr1, tr2, tr3 = [op.attrs['transpose_b'] for op in [y1.op.inputs[0].op, y2.op.inputs[0].op, y3.op.inputs[0].op]]
        if tr1 != tr2 or tr2 != tr3 or tr1 != tr3:
            return None

        if not tr1 and same_list(c1.shape[:-1], c2.shape[:-1]) and same_list(c2.shape[:-1], c3.shape[:-1]):
            if len(b1.shape) == len(b2.shape) == len(b3.shape) == 1:
                if b1.shape[0] == c1.shape[-1] and b2.shape[0] == c2.shape[-1] and b3.shape[0] == c3.shape[-1]:
                    c = ops.concat([c1, c2, c3], axis=-1)
                    b = ops.concat([b1, b2, b3], axis=-1)
                    y = ops.matmul(x, c) + b
                    # pylint: disable=unbalanced-tuple-unpacking
                    new_y1, new_y2, new_y3 = ops.split(
                        y, axis=-1, parts_or_sections=[y1.shape[-1], y2.shape[-1], y3.shape[-1]]
                    )
                    return [new_y1, new_y2, new_y3]

        elif (
            tr1
            and same_list(c1.shape[:-2] + (c1.shape[-1],), c2.shape[:-2] + (c2.shape[-1],))
            and same_list(c2.shape[:-2] + (c2.shape[-1],), c3.shape[:-2] + (c3.shape[-1],))
        ):
            if len(b1.shape) == len(b2.shape) == len(b3.shape) == 1:
                if b1.shape[0] == c1.shape[-2] and b2.shape[0] == c2.shape[-2] and b3.shape[0] == c3.shape[-2]:
                    c = ops.concat([c1, c2, c3], axis=-2)
                    b = ops.concat([b1, b2, b3], axis=-1)
                    y = ops.matmul_nt(x, c) + b
                    new_y1, new_y2, new_y3 = ops.split(
                        y, axis=-1, parts_or_sections=[y1.shape[-1], y2.shape[-1], y3.shape[-1]]
                    )
                    return [new_y1, new_y2, new_y3]

        return None


# Given matrices A of shape (m, k) and B of shape (n, k)
# We want to convert the following pattern
# B_transposed = ops.transpose(B)
# C = ops.matmul(A, B_transposed)
# into
# C = ops.matmul_nt(A, B)
class MatmulTransposeRewriteRule(SubgraphRewriteRule):
    def __init__(self):
        super().__init__('transpose(B) + matmul(A, B_transposed) ==> matmul_nt(A, B)')
        self.A = TensorPattern.tensor()
        self.B = TensorPattern.tensor()
        self.B_transposed = op_pattern(TransposeOp2D, [self.B])
        self.C = op_pattern(MatmulOp, [self.A, self.B_transposed])

    def source(self) -> List[TensorPattern]:
        return [self.C]

    def target(self, matched: MatchDict) -> Optional[List[Tensor]]:
        A, B, _, _ = [matched[t] for t in [self.A, self.B, self.B_transposed, self.C]]
        accepted_dtypes = [hidet.float16, hidet.bfloat16]
        if len(B.shape) != 2:
            return None
        if A.dtype not in accepted_dtypes or B.dtype not in accepted_dtypes:
            return None
        new_C = ops.matmul_nt(A, B)
        return [new_C]


# Related to the issue #935
# when the input A is a batched matrix, we want to flatten the batch dimensions and then reshape the result back
class BatchedMatmulFlattenRule(SubgraphRewriteRule):
    def __init__(self):
        super().__init__('matmul(A_ND, B_2D) ==> flatten(A_ND) + matmul + reshape')
        self.A = TensorPattern.tensor()
        self.B = TensorPattern.tensor()
        self.C = op_pattern(MatmulOp, [self.A, self.B])

    def source(self) -> List[TensorPattern]:
        return [self.C]

    def target(self, matched: MatchDict) -> Optional[List[Tensor]]:
        A, B, C = [matched[t] for t in [self.A, self.B, self.C]]

        # Check if A has at least 3 dimensions and B is a 2D tensor
        if len(A.shape) < 3 or len(B.shape) != 2:
            return None

        # Check if either A, B or C has more than one SymbolVar dimension in shape,
        # if so, skip it for now: since this has downstream impact on the construction of the dispatch table.
        # more specifically, say, we have a shape of [b, s, 256], if we want to reshape
        # it to [b * s, 256], then during the dispatch table, the expression `b * s` will be
        # treated as a new symbolic dimension aside from the original symbolic dimension `b` and `s`,
        # which will cause runtime failure, and fixing it seems to be trickier than I initially thought.
        # So for now, we just skip it.

        # TODO: If there is value supporting it:
        #  go back to the case where we have more than one SymbolVar dimension in shape

        a_shape, b_shape = A.shape, B.shape
        for tensor_shape in (a_shape, b_shape):
            if len([dim for dim in tensor_shape if not is_constant(dim)]) > 1:
                return None

        *batch_dims, k = A.shape  # All leading dimensions plus the last dimension k

        # Get transpose flag from the original operation
        transpose_b = C.op.attrs.get('transpose_b', False)

        # Check dimensions based on whether B is transposed
        if transpose_b:
            n, k2 = B.shape
            if k != k2:
                return None
        else:
            k2, n = B.shape
            if k != k2:
                return None

        # Calculate batch size
        batch_size = 1
        for dim in batch_dims:
            batch_size *= dim

        # Reshape A to [batch_size, k]
        A_reshaped = ops.reshape(A, [batch_size, k])

        if transpose_b:
            C_reshaped = ops.matmul_nt(A_reshaped, B)
        else:
            C_reshaped = ops.matmul(A_reshaped, B)

        # Reshape result back to [*batch_dims, n]
        output_shape = list(batch_dims) + [n]
        new_C = ops.reshape(C_reshaped, output_shape)

        return [new_C]


@initialize()
def matmul_patterns():
    register_rewrite_rule(MatmulTransposeRewriteRule())
    register_rewrite_rule(BatchedMatmulFlattenRule())
    # The following pattern matching passes were temporarily disabled earlier
    # register_rewrite_rule(ThreeMatmulBiasFusionPattern())
    # register_rewrite_rule(ThreeMatmulFusionPattern())
    # register_rewrite_rule(TwoMatmulFusionPattern())
