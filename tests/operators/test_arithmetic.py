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
import hidet
import numpy as np
from hidet import ops


def check_binary(a_shape, b_shape, dtype, op, hidet_op=None, a_positive=False, b_positive=False):
    a = np.random.rand(*a_shape).astype(dtype)
    b = np.random.rand(*b_shape).astype(dtype)
    a = np.abs(a) if a_positive else a
    b = np.abs(b) if b_positive else b
    numpy_c = op(a, b)
    if hidet_op is None:
        hidet_op = op
    hidet_c = hidet_op(hidet.asarray(a).cuda(), hidet.asarray(b).cuda()).cpu().numpy()
    np.testing.assert_allclose(actual=hidet_c, desired=numpy_c, atol=1e-5, rtol=1e-5)


def check_unary(shape, dtype, numpy_op, hidet_op, positive=False):
    a = np.random.rand(*shape).astype(dtype)
    a = np.abs(a) if positive else a
    numpy_b = numpy_op(a)
    hidet_b = hidet_op(hidet.asarray(a).cuda()).cpu().numpy()
    np.testing.assert_allclose(actual=hidet_b, desired=numpy_b, atol=1e-5, rtol=1e-5)


binary_op_shapes = [[[1], [200]], [[100, 200], [1, 200]], [[200, 1], [200]]]

unary_op_shapes = [[1], [100], [200]]


@pytest.mark.parametrize("a_shape, b_shape", binary_op_shapes)
def test_add(a_shape, b_shape):
    check_binary(a_shape, b_shape, np.float32, lambda a, b: a + b)


@pytest.mark.parametrize("a_shape, b_shape", binary_op_shapes)
def test_sub(a_shape, b_shape):
    check_binary(a_shape, b_shape, np.float32, lambda a, b: a - b)


@pytest.mark.parametrize("a_shape, b_shape", binary_op_shapes)
def test_multiply(a_shape, b_shape):
    check_binary(a_shape, b_shape, np.float32, lambda a, b: a * b)


@pytest.mark.parametrize("a_shape, b_shape", binary_op_shapes)
def test_divide(a_shape, b_shape):
    check_binary(a_shape, b_shape, np.float32, lambda a, b: a / b)


@pytest.mark.parametrize("a_shape, b_shape", binary_op_shapes)
def test_pow(a_shape, b_shape):
    check_binary(a_shape, b_shape, np.float32, np.power, ops.pow, a_positive=True)


@pytest.mark.parametrize("shape", unary_op_shapes)
def test_sqrt(shape):
    check_unary(shape, np.float32, np.sqrt, ops.sqrt, positive=True)


@pytest.mark.parametrize("shape", unary_op_shapes)
def test_tanh(shape):
    check_unary(shape, np.float32, np.tanh, ops.tanh)


@pytest.mark.parametrize("shape", unary_op_shapes)
def test_erf(shape):
    check_unary(shape, np.float32, np.tanh, ops.tanh)


@pytest.mark.parametrize("shape", unary_op_shapes)
def test_rsqrt(shape):
    check_unary(shape, np.float32, lambda v: np.reciprocal(np.sqrt(v)), ops.rsqrt, positive=True)


@pytest.mark.parametrize("shape", unary_op_shapes)
def test_neg(shape):
    check_unary(shape, np.float32, np.negative, ops.negative)


@pytest.mark.parametrize("shape", unary_op_shapes)
def test_abs(shape):
    check_unary(shape, np.float32, np.absolute, ops.abs)


@pytest.mark.parametrize("a_shape, b_shape", binary_op_shapes)
def test_rightshift(a_shape, b_shape):
    check_binary(a_shape, b_shape, np.uint32, np.right_shift, ops.bitwise_right_shift)


@pytest.mark.parametrize("a_shape, b_shape", binary_op_shapes)
def test_leftshift(a_shape, b_shape):
    check_binary(a_shape, b_shape, np.uint32, np.left_shift, ops.bitwise_left_shift)


@pytest.mark.parametrize("a_shape, b_shape", binary_op_shapes)
def test_bitwise_and(a_shape, b_shape):
    check_binary(a_shape, b_shape, np.int32, np.bitwise_and, ops.bitwise_and)


@pytest.mark.parametrize("a_shape", unary_op_shapes)
def test_bitwise_not(a_shape):
    check_unary(a_shape, np.int32, np.invert, ops.bitwise_invert)


@pytest.mark.parametrize("a_shape, b_shape", binary_op_shapes)
def test_bitwise_or(a_shape, b_shape):
    check_binary(a_shape, b_shape, np.int32, np.bitwise_or, ops.bitwise_or)


@pytest.mark.parametrize("a_shape, b_shape", binary_op_shapes)
def test_bitwise_xor(a_shape, b_shape):
    check_binary(a_shape, b_shape, np.int32, np.bitwise_xor, ops.bitwise_xor)


@pytest.mark.parametrize("a_shape, b_shape", binary_op_shapes)
def test_minimum(a_shape, b_shape):
    check_binary(a_shape, b_shape, np.int32, np.minimum, ops.minimum)


@pytest.mark.parametrize("a_shape, b_shape", binary_op_shapes)
def test_maximum(a_shape, b_shape):
    check_binary(a_shape, b_shape, np.int32, np.maximum, ops.maximum)


@pytest.mark.parametrize("a_shape", unary_op_shapes)
def test_ceil(a_shape):
    check_unary(a_shape, np.float32, np.ceil, ops.ceil)


@pytest.mark.parametrize("a_shape", unary_op_shapes)
def test_cast_from_fp16(a_shape):
    check_unary(a_shape, np.float16, np.int8, lambda x: ops.cast(x, "int8"))
    check_unary(a_shape, np.float16, np.uint8, lambda x: ops.cast(x, "uint8"))

    check_unary(a_shape, np.float16, np.int16, lambda x: ops.cast(x, "int16"))
    check_unary(a_shape, np.float16, np.uint16, lambda x: ops.cast(x, "uint16"))

    check_unary(a_shape, np.float16, np.int32, lambda x: ops.cast(x, "int32"))
    check_unary(a_shape, np.float16, np.uint32, lambda x: ops.cast(x, "uint32"))

    check_unary(a_shape, np.float16, np.int64, lambda x: ops.cast(x, "int64"))
    check_unary(a_shape, np.float16, np.uint64, lambda x: ops.cast(x, "uint64"))


if __name__ == '__main__':
    pytest.main([__file__])
