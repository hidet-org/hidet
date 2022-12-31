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
from hidet.ir import Expr, expr
from .arithmetic import BinaryElementwiseOp, UnaryElementwiseOp
from .utils import Tensor


class EqualOp(BinaryElementwiseOp):
    def __init__(self, x: Tensor, y: Tensor):
        super().__init__(x, y, lambda a, b: expr.Equal(a, b), name='eq')


class NotEqualOp(BinaryElementwiseOp):
    def __init__(self, x: Tensor, y: Tensor):
        super().__init__(x, y, lambda a, b: expr.NotEqual(a, b), name='ne')


class LessOp(BinaryElementwiseOp):
    def __init__(self, x: Tensor, y: Tensor):
        super().__init__(x, y, lambda a, b: a < b, name='lt')


class GreaterOp(BinaryElementwiseOp):
    def __init__(self, x: Tensor, y: Tensor):
        super().__init__(x, y, lambda a, b: a > b, name='gt')


class LessEqualOp(BinaryElementwiseOp):
    def __init__(self, x: Tensor, y: Tensor):
        super().__init__(x, y, lambda a, b: a <= b, name='le')


class GreaterEqualOp(BinaryElementwiseOp):
    def __init__(self, x: Tensor, y: Tensor):
        super().__init__(x, y, lambda a, b: a >= b, name='ge')


class LogicalNotOp(UnaryElementwiseOp):
    def __init__(self, x: Tensor):
        super().__init__(x, lambda a: expr.LogicalNot(a), name='not')


class LogicalAndOp(BinaryElementwiseOp):
    def __init__(self, x: Tensor, y: Tensor):
        super().__init__(x, y, lambda a, b: expr.LogicalAnd(a, b), name='and')


class LogicalOrOp(BinaryElementwiseOp):
    def __init__(self, x: Tensor, y: Tensor):
        super().__init__(x, y, lambda a, b: expr.LogicalOr(a, b), name='or')


class LogicalXorOp(BinaryElementwiseOp):
    def __init__(self, x: Tensor, y: Tensor):
        def expr_logical_xor(a: Expr, b: Expr) -> Expr:
            x = expr.LogicalAnd(a, expr.LogicalNot(b))
            y = expr.LogicalAnd(expr.LogicalNot(a), b)
            return expr.LogicalOr(x, y)

        super().__init__(x, y, lambda a, b: expr_logical_xor(a, b), name='xor')


def equal(x: Tensor, y: Tensor) -> Tensor:
    return EqualOp(x, y).get_output(0)


def not_equal(x: Tensor, y: Tensor) -> Tensor:
    return NotEqualOp(x, y).get_output(0)


def less(x: Tensor, y: Tensor) -> Tensor:
    return LessOp(x, y).get_output(0)


def greater(x: Tensor, y: Tensor) -> Tensor:
    return GreaterOp(x, y).get_output(0)


def less_equal(x: Tensor, y: Tensor) -> Tensor:
    return LessEqualOp(x, y).get_output(0)


def greater_equal(x: Tensor, y: Tensor) -> Tensor:
    return GreaterEqualOp(x, y).get_output(0)


def logical_not(x: Tensor) -> Tensor:
    return LogicalNotOp(x).get_output(0)


def logical_or(x: Tensor, y: Tensor) -> Tensor:
    return LogicalOrOp(x, y).get_output(0)


def logical_and(x: Tensor, y: Tensor) -> Tensor:
    return LogicalAndOp(x, y).get_output(0)


def logical_xor(x: Tensor, y: Tensor) -> Tensor:
    return LogicalXorOp(x, y).get_output(0)
