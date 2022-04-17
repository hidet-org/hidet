from hidet.ir.primitives.func import cuda_max, cuda_exp

from ..basic.arithmatic import UnaryElementwiseOp
from ..common import Tensor


class ReluOp(UnaryElementwiseOp):
    def __init__(self, x):
        super().__init__(x, op=lambda v: cuda_max(v, 0.0), name='relu')


class SigmoidOp(UnaryElementwiseOp):
    def __init__(self, x):
        super().__init__(x, op=lambda v: 1.0 / (1.0 + cuda_exp(-v)), name='sigmoid')


def relu(x) -> Tensor:
    return ReluOp(x).get_output(0)


def sigmoid(x: Tensor) -> Tensor:
    return SigmoidOp(x).get_output(0)


