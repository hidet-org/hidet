from typing import List, Optional, Dict, Union

from hidet.tos import ops
from hidet.tos.ir.graph import Operator, Tensor
from hidet.tos.ops.definitions.matmul import MatmulOp
from .base import GraphPattern, TensorPattern, OperatorPattern, op_pattern


class MatmulRightScalePattern(GraphPattern):
    def __init__(self):
        super().__init__('matmul(x, c1) * c2 => matmul(x, c1 * c2)')
        self.x = TensorPattern.tensor()
        self.c1 = TensorPattern.tensor(is_const=True)
        self.c2 = TensorPattern.tensor(is_const=True)
        self.y = op_pattern(MatmulOp, [self.x, self.c1]) * self.c2

    def source(self) -> TensorPattern:
        return self.y

    def target(self, matched: Dict[Union[TensorPattern, OperatorPattern], Union[Tensor, Operator]]) -> Optional[Tensor]:
        x, c1, c2, y = [matched[v] for v in [self.x, self.c1, self.c2, self.y]]
        if len(c2.shape) >= 2 and c2.shape[-2] != 1:  # c2 should have shape [., 1, .]
            return None
        return ops.batched_matmul(x, c1 * c2)


class MatmulLeftScalePattern(GraphPattern):
    def __init__(self):
        super().__init__('matmul(c1, x) * c2 => matmul(c1 * c2, x)')
        self.c1 = TensorPattern.tensor(is_const=True)
        self.x = TensorPattern.tensor(is_symbolic=True)
        self.c2 = TensorPattern.tensor(is_const=True)
        self.y = op_pattern(MatmulOp, [self.c1, self.x]) * self.c2

    def source(self) -> TensorPattern:
        return self.y

    def target(self, matched: Dict[Union[TensorPattern, OperatorPattern], Union[Tensor, Operator]]) -> Optional[Tensor]:
        c1, x, c2, y = [matched[v] for v in [self.c1, self.x, self.c2, self.y]]
        if len(c2.shape) > 0 and c2.shape[-1] != 1:  # c2 should have shape [., ., 1]
            return None
        return ops.batched_matmul(c1 * c2, x)


def matmul_patterns() -> List[GraphPattern]:
    return [
        MatmulRightScalePattern(),
        MatmulLeftScalePattern(),
    ]
