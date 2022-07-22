from hidet.ir import expr
from .arithmatic import BinaryElementwiseOp
from .utils import Tensor


class EqualOp(BinaryElementwiseOp):
    def __init__(self, x: Tensor, y: Tensor):
        super().__init__(x, y, lambda a, b: expr.Equal(a, b), name='eq')


class LessOp(BinaryElementwiseOp):
    def __init__(self, x: Tensor, y: Tensor):
        super().__init__(x, y, lambda a, b: a < b, name='lt')


class GreaterOp(BinaryElementwiseOp):
    def __init__(self, x: Tensor, y: Tensor):
        super().__init__(x, y, lambda a, b: a > b, name='gt')


class LessOrEqual(BinaryElementwiseOp):
    def __init__(self, x: Tensor, y: Tensor):
        super().__init__(x, y, lambda a, b: a <= b, name='le')


class GreaterOrEqual(BinaryElementwiseOp):
    def __init__(self, x: Tensor, y: Tensor):
        super().__init__(x, y, lambda a, b: a >= b, name='ge')


def equal(x: Tensor, y: Tensor) -> Tensor:
    if x.dtype != y.dtype:
        raise ValueError('Can only compare tensors with the same dtype, but got {} and {}'.format(x.dtype, y.dtype))
    return EqualOp(x, y).get_output(0)


def less_than(x: Tensor, y: Tensor) -> Tensor:
    return LessOp(x, y).get_output(0)


def greater_than(x: Tensor, y: Tensor) -> Tensor:
    return GreaterOp(x, y).get_output(0)


def less_or_equal(x: Tensor, y: Tensor) -> Tensor:
    return LessOrEqual(x, y).get_output(0)


def greater_or_equal(x: Tensor, y: Tensor) -> Tensor:
    return GreaterOrEqual(x, y).get_output(0)
