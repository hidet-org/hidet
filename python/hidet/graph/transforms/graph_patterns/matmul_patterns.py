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
from hidet.graph.ops.matmul import MatmulOp
from hidet.utils import same_list, initialize

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


@initialize()
def matmul_patterns():
    pass
    # register_rewrite_rule(ThreeMatmulBiasFusionPattern())
    # register_rewrite_rule(ThreeMatmulFusionPattern())
    # register_rewrite_rule(TwoMatmulFusionPattern())
