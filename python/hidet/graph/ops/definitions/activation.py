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
from typing import Optional
import math
from hidet.ir import primitives as prim
from hidet.ir.expr import if_then_else, BitwiseAnd
from .utils import Tensor, Operator, normalize_dim, input_like
from .arithmetic import UnaryElementwiseOp, BinaryElementwiseOp
from .softmax import SoftmaxTask


class ReluOp(UnaryElementwiseOp):
    def __init__(self, x: Tensor):
        super().__init__(x, op=lambda v: prim.max(v, x.dtype.zero), name='relu')


class LeakyReluOp(UnaryElementwiseOp):
    def __init__(self, x: Tensor, alpha):
        super().__init__(
            x, op=lambda v: if_then_else(v >= 0, v, v * x.dtype(alpha)), name='leaky_relu', attributes={'alpha': alpha}
        )


class SigmoidOp(UnaryElementwiseOp):
    def __init__(self, x: Tensor):
        super().__init__(x, op=lambda v: x.dtype(1.0) / (x.dtype(1.0) + prim.exp(-v)), name='sigmoid')


class HardSigmoidOp(UnaryElementwiseOp):
    def __init__(self, x: Tensor):
        super().__init__(
            x,
            op=lambda v: if_then_else(
                v <= x.dtype(-3),
                x.dtype.zero,
                if_then_else(v >= x.dtype(3), x.dtype.one, v / x.dtype(6) + x.dtype(0.5)),
            ),
            name='hardsigmoid',
        )


class ClipOp(UnaryElementwiseOp):
    def __init__(self, x: Tensor, min_val: Optional[float] = None, max_val: Optional[float] = None):
        def op(v):
            if min_val is not None:
                v = prim.max(v, x.dtype(min_val))
            if max_val is not None:
                v = prim.min(v, x.dtype(max_val))
            return v

        super().__init__(x, op=op, name='clip', attributes={'min_val': min_val, 'max_val': max_val})


class GeluOp(UnaryElementwiseOp):
    def __init__(self, x: Tensor):
        super().__init__(
            x, op=lambda v: x.dtype(0.5) * v * (x.dtype.one + prim.erf(v * x.dtype(1 / math.sqrt(2)))), name='gelu'
        )


class SiluOp(UnaryElementwiseOp):
    def __init__(self, x: Tensor):
        super().__init__(x, op=lambda v: v * (x.dtype(1.0) / (x.dtype(1.0) + prim.exp(-v))), name='silu')


class PReluOp(BinaryElementwiseOp):
    def __init__(self, x, slope):
        super().__init__(x, slope, op=lambda a, b: if_then_else(a >= 0, a, a * b), name='prelu')


class HardSwishOp(UnaryElementwiseOp):
    def __init__(self, x: Tensor):
        super().__init__(
            x,
            op=lambda v: if_then_else(
                v <= x.dtype(-3), x.dtype.zero, if_then_else(v >= x.dtype(3), v, (v * (v + x.dtype(3))) / x.dtype(6))
            ),
            name='hardswish',
        )


class ThresholdOp(UnaryElementwiseOp):
    def __init__(self, x: Tensor, threshold_val: float, value: float) -> Tensor:
        super().__init__(x, op=lambda v: if_then_else(v > x.dtype(threshold_val), v, x.dtype(value)), name='threshold')


class HardTanhOp(UnaryElementwiseOp):
    def __init__(self, x: Tensor, min_val: float = -1.0, max_val: float = 1.0) -> Tensor:
        super().__init__(x, op=lambda v: prim.min(x.dtype(max_val), prim.max(x.dtype(min_val), v)), name='hardtanh')


class EluOp(UnaryElementwiseOp):
    def __init__(self, x: Tensor, alpha: float = 1.0) -> Tensor:
        super().__init__(
            x, op=lambda v: if_then_else(v > 0, v, x.dtype(alpha) * (prim.exp(v) - x.dtype(1.0))), name='elu'
        )


class SeluOp(UnaryElementwiseOp):
    def __init__(
        self,
        x: Tensor,
        alpha: float = 1.6732632423543772848170429916717,
        scale: float = 1.0507009873554804934193349852946,
    ) -> Tensor:
        super().__init__(
            x,
            op=lambda v: x.dtype(scale)
            * (prim.max(x.dtype(0.0), v) + prim.min(x.dtype(0.0), x.dtype(alpha) * (prim.exp(v) - x.dtype(-1.0)))),
            name='selu',
        )


class CeluOp(UnaryElementwiseOp):
    def __init__(self, x: Tensor, alpha: float = 1.0) -> Tensor:
        super().__init__(
            x,
            op=lambda v: prim.max(x.dtype(0.0), v)
            + prim.min(x.dtype(0.0), x.dtype(alpha) * (prim.exp(v / x.dtype(alpha)) - x.dtype(1.0))),
            name='celu',
        )


class LogSigmoidOp(UnaryElementwiseOp):
    def __init__(self, x: Tensor):
        super().__init__(x, op=lambda v: -(prim.log(x.dtype(1.0) + prim.exp(-v))), name='logsigmoid')


class HardShrinkOp(UnaryElementwiseOp):
    def __init__(self, x: Tensor, lambda_val: float = 0.5):
        super().__init__(
            x,
            op=lambda v: if_then_else(BitwiseAnd(v >= x.dtype(-lambda_val), v <= x.dtype(lambda_val)), x.dtype(0), v),
            name='hardshrink',
        )


class TanhShrinkOp(UnaryElementwiseOp):
    def __init__(self, x: Tensor):
        super().__init__(
            x, op=lambda v: v - (prim.exp(v) - prim.exp(-v)) / (prim.exp(v) + prim.exp(-v)), name='tanhshrink'
        )


class SoftSignOp(UnaryElementwiseOp):
    def __init__(self, x: Tensor):
        super().__init__(x, op=lambda v: v / (x.dtype(1.0) + prim.abs(v)), name='softsign')


class SoftPlusOp(UnaryElementwiseOp):
    def __init__(self, x: Tensor, beta: int = 1, threshold_val: int = 20):
        super().__init__(
            x,
            op=lambda v: if_then_else(
                v * x.dtype(beta) <= x.dtype(threshold_val),
                (x.dtype(1.0 / beta)) * prim.log(x.dtype(1.0) + prim.exp(x.dtype(beta) * v)),
                v,
            ),
            name='softplus',
        )


class SoftShrinkOp(UnaryElementwiseOp):
    def __init__(self, x: Tensor, lambda_val: float = 0.5):
        super().__init__(
            x,
            op=lambda v: if_then_else(
                v > x.dtype(lambda_val),
                v - x.dtype(lambda_val),
                if_then_else(v < x.dtype(-lambda_val), v + x.dtype(lambda_val), x.dtype(0.0)),
            ),
            name='softshrink',
        )


class SoftmaxOp(Operator):
    def __init__(self, x: Tensor, axis: int = 1):
        axis = normalize_dim(axis, len(x.shape))
        super().__init__(inputs=[x], attributes={'axis': axis}, task=SoftmaxTask(input_like(x, 'x'), axis))


def relu(x) -> Tensor:
    return ReluOp(x).get_output(0)


def leaky_relu(x: Tensor, alpha: float) -> Tensor:
    return LeakyReluOp(x, alpha).get_output(0)


def sigmoid(x: Tensor) -> Tensor:
    return SigmoidOp(x).get_output(0)


def hardsigmoid(x: Tensor) -> Tensor:
    return HardSigmoidOp(x).get_output(0)


def clip(x: Tensor, min_val: Optional[float], max_val: Optional[float]) -> Tensor:
    return ClipOp(x, min_val, max_val).get_output(0)


def relu6(x: Tensor) -> Tensor:
    return clip(x, 0.0, 6.0)


def gelu(x: Tensor) -> Tensor:
    return GeluOp(x).get_output(0)


def silu(x: Tensor) -> Tensor:
    return SiluOp(x).get_output(0)


def prelu(x: Tensor, slope: Tensor) -> Tensor:
    return PReluOp(x, slope).get_output(0)


def hardswish(x: Tensor) -> Tensor:
    return HardSwishOp(x).get_output(0)


def threshold(x: Tensor, threshold_val: float, value: float) -> Tensor:
    return ThresholdOp(x, threshold_val, value).get_output(0)


def hardtanh(x: Tensor, min_val: float, max_val: float) -> Tensor:
    return HardTanhOp(x, min_val, max_val).get_output(0)


def elu(x: Tensor, alpha: float) -> Tensor:
    return EluOp(x, alpha).get_output(0)


def selu(x: Tensor, alpha: float, scale: float) -> Tensor:
    return SeluOp(x, alpha, scale).get_output(0)


def celu(x: Tensor, alpha: float) -> Tensor:
    return CeluOp(x, alpha).get_output(0)


def logsigmoid(x: Tensor) -> Tensor:
    return LogSigmoidOp(x).get_output(0)


def hardshrink(x: Tensor, lambda_val: float) -> Tensor:
    return HardShrinkOp(x, lambda_val).get_output(0)


def tanhshrink(x: Tensor) -> Tensor:
    return TanhShrinkOp(x).get_output(0)


def softsign(x: Tensor) -> Tensor:
    return SoftSignOp(x).get_output(0)


def softplus(x: Tensor, beta: int, threshold_val: int) -> Tensor:
    return SoftPlusOp(x, beta, threshold_val).get_output(0)


def softshrink(x: Tensor, lambda_val: float) -> Tensor:
    return SoftShrinkOp(x, lambda_val).get_output(0)


def softmax(x: Tensor, axis=1) -> Tensor:
    return SoftmaxOp(x, axis).get_output(0)


def softmin(x: Tensor, axis: int) -> Tensor:
    return SoftmaxOp(-x, axis).get_output(0)
