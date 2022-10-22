from typing import List, Optional, Dict, Union

from hidet.graph import ops
from hidet.graph.ir.flow_graph import Tensor
from hidet.graph.ops.definitions.matmul import MatmulOp
from .base import GraphPattern, TensorPattern, MatchDict, op_pattern
from hidet.utils import same_list


class TwoMatmulFusionPattern(GraphPattern):
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
        x, c1, c2, y1, y2 = [matched[t] for t in [self.x, self.c1, self.c2, self.y1, self.y2]]
        if 2 <= len(c1.shape) == len(c2.shape) and same_list(c1.shape[:-1], c2.shape[:-1]):
            c = ops.concat([c1, c2], axis=-1)
            y = ops.matmul(x, c)
            new_y1, new_y2 = ops.split(y, axis=-1, parts=[c1.shape[-1], c2.shape[-1]])
            return [new_y1, new_y2]
        else:
            return None


class ThreeMatmulFusionPattern(GraphPattern):
    def __init__(self):
        super().__init__('matmul(x, c1)|matmul(x, c2)|matmul(x, c3) ==> matmul(x, concat(c1, c2, c3)) followed by split')
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
        x, c1, c2, c3, y1, y2, y3 = [matched[t] for t in [self.x, self.c1, self.c2, self.c3, self.y1, self.y2, self.y3]]
        if len(c1.shape) == len(c2.shape) == len(c3.shape) >= 2 and same_list(c1.shape[:-1], c2.shape[:-1]) and same_list(c2.shape[:-1], c3.shape[:-1]):
            c = ops.concat([c1, c2, c3], axis=-1)
            y = ops.matmul(x, c)
            new_y1, new_y2, new_y3 = ops.split(y, axis=-1, parts=[c1.shape[-1], c2.shape[-1], c3.shape[-1]])
            return [new_y1, new_y2, new_y3]
        else:
            return None


class ThreeMatmulBiasFusionPattern(GraphPattern):
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
        x, c1, c2, c3, b1, b2, b3, y1, y2, y3 = [matched[t] for t in [self.x, self.c1, self.c2, self.c3, self.b1, self.b2, self.b3, self.y1, self.y2, self.y3]]
        if len(c1.shape) == len(c2.shape) == len(c3.shape) >= 2:
            if same_list(c1.shape[:-1], c2.shape[:-1], use_equal=True) and same_list(c2.shape[:-1], c3.shape[:-1], use_equal=True):
                if len(b1.shape) == len(b2.shape) == len(b3.shape) == 1:
                    if b1.shape[0] == c1.shape[-1] and b2.shape[0] == c2.shape[-1] and b3.shape[0] == c3.shape[-1]:
                        c = ops.concat([c1, c2, c3], axis=-1)
                        b = ops.concat([b1, b2, b3], axis=-1)
                        y = ops.matmul(x, c) + b
                        new_y1, new_y2, new_y3 = ops.split(y, axis=-1, parts=[y1.shape[-1], y2.shape[-1], y3.shape[-1]])
                        return [new_y1, new_y2, new_y3]
        return None


def matmul_patterns() -> List[GraphPattern]:
    return [
        ThreeMatmulBiasFusionPattern(),
        ThreeMatmulFusionPattern(),
        TwoMatmulFusionPattern(),
    ]
