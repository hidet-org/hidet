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
import pytest
import operator
import hidet
from hidet import int32, boolean, float16, bfloat16, float32, int8, uint16
from hidet.ir.expr import Constant


def check(a: Constant, b: Constant):
    assert isinstance(a, Constant)
    assert isinstance(b, Constant)
    assert a.type == b.type
    assert a.value == b.value


@pytest.mark.parametrize(
    'op, result',
    [
        # arithmetic
        [operator.add, int32(8)],
        [operator.sub, int32(2)],
        [operator.mul, int32(15)],
        [operator.truediv, int32(1)],  # hidet treat a / b and a // b as the same semantics as C language
        [operator.floordiv, int32(1)],
        [operator.mod, int32(2)],
        # comparison
        [operator.le, boolean(False)],
        [operator.lt, boolean(False)],
        [operator.ge, boolean(True)],
        [operator.gt, boolean(True)],
        [operator.eq, boolean(False)],
        [operator.ne, boolean(True)],
        # bitwise
        [operator.and_, int32(5 & 3)],
        [operator.or_, int32(5 | 3)],
        [operator.xor, int32(5 ^ 3)],
        [operator.lshift, int32(5 << 3)],
        [operator.rshift, int32(5 >> 3)],
    ],
)
def test_arithmetic_binary_op(op, result):
    a = hidet.ir.dtypes.int32(5)
    b = hidet.ir.dtypes.int32(3)
    check(op(a, b), result)


@pytest.mark.parametrize('op, result', [[operator.neg, int32(-5)]])
def test_arithmetic_unary_op(op, result):
    a = hidet.ir.dtypes.int32(5)
    check(op(a), result)


def test_expr_const_fold():
    a = int32(1)
    b = float32(2.0)
    c = bfloat16(3.0)
    d = int8(4)
    e = uint16(5)
    f = float16(6.0)

    check(a + b, float32(3.0))
    check(a + c, bfloat16(4.0))
    check(a + f, float16(7.0))
    check(a + d, int32(5))
    check(a + e, int32(6))
    check(b + c, float32(5.0))
    check(b + f, float32(8.0))
    check(b + d, float32(6.0))
    check(b + e, float32(7.0))
    check(c + d, bfloat16(7.0))
    check(c + e, bfloat16(8.0))
    check(d + e, int32(9))
    check(f + d, float16(10.0))
    check(f + e, float16(11.0))
