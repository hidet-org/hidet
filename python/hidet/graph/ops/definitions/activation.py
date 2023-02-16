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
from hidet.ir.expr import if_then_else
from .utils import Tensor
from .arithmetic import UnaryElementwiseOp, BinaryElementwiseOp


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
        super().__init__(x, op=lambda v: x.dtype(1.0) / (x.dtype.one + prim.exp(-v)), name='sigmoid')


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
        dtype = x.dtype
        super().__init__(
            x=x, op=lambda v: dtype(0.5) * v * (dtype.one + prim.erf(v * dtype(1 / math.sqrt(2)))), name='gelu'
        )


class PReluOp(BinaryElementwiseOp):
    def __init__(self, x, slope):
        super().__init__(x, slope, op=lambda a, b: if_then_else(a >= 0, a, a * b), name='prelu')


def relu(x) -> Tensor:
    return ReluOp(x).get_output(0)


def leaky_relu(x: Tensor, alpha: float) -> Tensor:
    return LeakyReluOp(x, alpha).get_output(0)


def sigmoid(x: Tensor) -> Tensor:
    return SigmoidOp(x).get_output(0)


def clip(x: Tensor, min_val: Optional[float], max_val: Optional[float]) -> Tensor:
    return ClipOp(x, min_val, max_val).get_output(0)


def relu6(x: Tensor) -> Tensor:
    return clip(x, 0.0, 6.0)


def gelu(x: Tensor) -> Tensor:
    return GeluOp(x).get_output(0)


def prelu(x: Tensor, slope: Tensor) -> Tensor:
    return PReluOp(x, slope).get_output(0)
