from hidet.ir.primitives.func import cuda_max

from ..basic.arithmatic import UnaryElementwiseOp
from ..common import Tensor


class ReluOp(UnaryElementwiseOp):
    def __init__(self, x):
        super().__init__(x, op=lambda v: cuda_max(v, 0.0), name='relu')


def relu(input) -> Tensor:
    return ReluOp(input).get_output(0)

