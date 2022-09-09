from typing import Optional
import math
from hidet.ir import primitives as prim, convert
from hidet.ir.expr import const_like

from .utils import Tensor
from .arithmatic import UnaryElementwiseOp, erf, tanh, cube


class ReluOp(UnaryElementwiseOp):
    def __init__(self, x):
        super().__init__(x, op=lambda v: prim.max(v, const_like(0.0, v)), name='relu')


class SigmoidOp(UnaryElementwiseOp):
    def __init__(self, x):
        super().__init__(x, op=lambda v: const_like(1.0, v) / (const_like(1.0, v) + prim.exp(-v)), name='sigmoid')


class ClipOp(UnaryElementwiseOp):
    def __init__(self, x, min_val: Optional[float] = None, max_val: Optional[float] = None):
        def op(v):
            if min_val is not None:
                v = prim.max(v, const_like(min_val, v))
            if max_val is not None:
                v = prim.min(v, const_like(max_val, v))
            return v

        super().__init__(x, op=op, name='clip')


class GeluOp(UnaryElementwiseOp):
    def __init__(self, x):
        super().__init__(x, op=lambda v: const_like(0.5, v) * v * (const_like(1.0, v) + prim.erf(v * const_like(1 / math.sqrt(2), v))), name='gelu')


def relu(x) -> Tensor:
    return ReluOp(x).get_output(0)


def sigmoid(x: Tensor) -> Tensor:
    return SigmoidOp(x).get_output(0)


def clip(x: Tensor, min_val: Optional[float], max_val: Optional[float]) -> Tensor:
    return ClipOp(x, min_val, max_val).get_output(0)


def relu6(x: Tensor) -> Tensor:
    return clip(x, 0.0, 6.0)


def gelu(x: Tensor) -> Tensor:
    return GeluOp(x).get_output(0)
