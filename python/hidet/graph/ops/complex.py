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
# pylint: disable=redefined-outer-name
from hidet.ir import primitives
from .utils import Tensor
from .arithmetic import UnaryElementwiseOp, BinaryElementwiseOp


class RealOperator(UnaryElementwiseOp):
    def __init__(self, x: Tensor):
        if not x.dtype.is_complex():
            raise ValueError('input tensor must be complex')
        super().__init__(x, op=lambda v: primitives.real(v), name='real')


class ImagOperator(UnaryElementwiseOp):
    def __init__(self, x: Tensor):
        if not x.dtype.is_complex():
            raise ValueError('input tensor must be complex')
        super().__init__(x, op=lambda v: primitives.imag(v), name='imag')


class ConjOperator(UnaryElementwiseOp):
    def __init__(self, x: Tensor):
        if not x.dtype.is_complex():
            raise ValueError('input tensor must be complex')
        super().__init__(x, op=lambda v: primitives.conj(v), name='conj')


class MakeComplexOperator(BinaryElementwiseOp):
    def __init__(self, real: Tensor, imag: Tensor):
        super().__init__(real, imag, op=lambda a, b: primitives.make_complex(a, b), name='make_complex')


def real(x: Tensor) -> Tensor:
    return RealOperator(x).outputs[0]


def imag(x: Tensor) -> Tensor:
    return ImagOperator(x).outputs[0]


def conj(x: Tensor) -> Tensor:
    return ConjOperator(x).outputs[0]


def make_complex(real: Tensor, imag: Tensor) -> Tensor:
    return MakeComplexOperator(real, imag).outputs[0]
