from typing import List, Optional, Dict, Union

from hidet.tos import ops
from hidet.tos.ir.graph import Operator, Tensor
from hidet.tos.ops.definitions.matmul import MatmulOp
from .base import GraphPattern, TensorPattern, OperatorPattern, op_pattern


class MatmulMultiplyPattern(GraphPattern):
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
        if c2.shape[1] != 1:  # c2 should have shape [., 1, .]
            return None
        return ops.batched_matmul(x, c1 * c2)


def matmul_patterns() -> List[GraphPattern]:
    return [
        MatmulMultiplyPattern()
    ]
