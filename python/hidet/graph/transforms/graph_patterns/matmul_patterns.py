from typing import List, Optional, Dict, Union

from hidet.graph import ops
from hidet.graph.ir.flow_graph import Operator, Tensor
from hidet.graph.ops.definitions.matmul import BatchMatmulOp
from .base import GraphPattern, TensorPattern, OperatorPattern, MatchDict, op_pattern


class MatmulRightScalePattern(GraphPattern):
    def __init__(self):
        super().__init__('matmul(x, c1) * c2 => matmul(x, c1 * c2)')
        self.x = TensorPattern.tensor()
        self.c1 = TensorPattern.tensor(is_const=True)
        self.c2 = TensorPattern.tensor(is_const=True)
        self.y = op_pattern(BatchMatmulOp, [self.x, self.c1]) * self.c2

    def source(self) -> List[TensorPattern]:
        return [self.y]

    def target(self, matched: MatchDict) -> Optional[List[Tensor]]:
        x, c1, c2, y = [matched[v] for v in [self.x, self.c1, self.c2, self.y]]
        if len(c2.shape) >= 2 and c2.shape[-2] != 1:  # c2 should have shape [., 1, .]
            return None
        return [ops.batch_matmul(x, c1 * c2)]


class MatmulLeftScalePattern(GraphPattern):
    def __init__(self):
        super().__init__('matmul(c1, x) * c2 ==> matmul(c1 * c2, x)')
        self.c1 = TensorPattern.tensor(is_const=True)
        self.x = TensorPattern.tensor(is_symbolic=True)
        self.c2 = TensorPattern.tensor(is_const=True)
        self.y = op_pattern(BatchMatmulOp, [self.c1, self.x]) * self.c2

    def source(self) -> List[TensorPattern]:
        return [self.y]

    def target(self, matched: MatchDict) -> Optional[List[Tensor]]:
        c1, x, c2, y = [matched[v] for v in [self.c1, self.x, self.c2, self.y]]
        if len(c2.shape) > 0 and c2.shape[-1] != 1:  # c2 should have shape [., ., 1]
            return None
        return [ops.batch_matmul(c1 * c2, x)]


class TwoMatmulFusionPattern(GraphPattern):
    def __init__(self):
        super().__init__('matmul(x, c1)|matmul(x, c2) ==> matmul(x, concat(c1, c2)) followed by split')
        self.x = TensorPattern.tensor(is_symbolic=True)
        self.c1 = TensorPattern.tensor(is_const=True)
        self.c2 = TensorPattern.tensor(is_const=True)
        self.y1 = op_pattern(BatchMatmulOp, [self.x, self.c1])
        self.y2 = op_pattern(BatchMatmulOp, [self.x, self.c2])

    def source(self) -> List[TensorPattern]:
        return [self.y1, self.y2]

    def target(self, matched: MatchDict) -> Optional[List[Tensor]]:
        x, c1, c2, y1, y2 = [matched[t] for t in [self.x, self.c1, self.c2, self.y1, self.y2]]
        c = ops.concat([c1, c2], axis=2)
        y = ops.batch_matmul(x, c)
        new_y1, new_y2 = ops.split(y, axis=2, parts=[y1.shape[2], y2.shape[2]])
        return [new_y1, new_y2]


class ThreeMatmulFusionPattern(GraphPattern):
    def __init__(self):
        super().__init__('matmul(x, c1)|matmul(x, c2)|matmul(x, c3) ==> matmul(x, concat(c1, c2, c3)) followed by split')
        self.x = TensorPattern.tensor(is_symbolic=True)
        self.c1 = TensorPattern.tensor(is_const=True)
        self.c2 = TensorPattern.tensor(is_const=True)
        self.c3 = TensorPattern.tensor(is_const=True)
        self.y1 = op_pattern(BatchMatmulOp, [self.x, self.c1])
        self.y2 = op_pattern(BatchMatmulOp, [self.x, self.c2])
        self.y3 = op_pattern(BatchMatmulOp, [self.x, self.c3])

    def source(self) -> List[TensorPattern]:
        return [self.y1, self.y2, self.y3]

    def target(self, matched: MatchDict) -> Optional[List[Tensor]]:
        x, c1, c2, c3, y1, y2, y3 = [matched[t] for t in [self.x, self.c1, self.c2, self.c3, self.y1, self.y2, self.y3]]
        c = ops.concat([c1, c2, c3], axis=2)
        y = ops.batch_matmul(x, c)
        new_y1, new_y2, new_y3 = ops.split(y, axis=2, parts=[y1.shape[2], y2.shape[2], y3.shape[2]])
        return [new_y1, new_y2, new_y3]


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
        self.y1 = op_pattern(BatchMatmulOp, [self.x, self.c1]) + self.b1
        self.y2 = op_pattern(BatchMatmulOp, [self.x, self.c2]) + self.b2
        self.y3 = op_pattern(BatchMatmulOp, [self.x, self.c3]) + self.b3

    def source(self) -> List[TensorPattern]:
        return [self.y1, self.y2, self.y3]

    def target(self, matched: MatchDict) -> Optional[List[Tensor]]:
        x, c1, c2, c3, b1, b2, b3, y1, y2, y3 = [matched[t] for t in [self.x, self.c1, self.c2, self.c3, self.b1, self.b2, self.b3, self.y1, self.y2, self.y3]]
        c = ops.concat([c1, c2, c3], axis=2)
        b = ops.concat([b1, b2, b3], axis=-1)
        y = ops.batch_matmul(x, c) + b
        new_y1, new_y2, new_y3 = ops.split(y, axis=2, parts=[y1.shape[2], y2.shape[2], y3.shape[2]])
        return [new_y1, new_y2, new_y3]


def matmul_patterns() -> List[GraphPattern]:
    return [
        MatmulRightScalePattern(),
        MatmulLeftScalePattern(),
        ThreeMatmulBiasFusionPattern(),
        ThreeMatmulFusionPattern(),
        TwoMatmulFusionPattern(),
    ]
