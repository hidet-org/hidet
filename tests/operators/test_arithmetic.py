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
import math
import pytest
import hidet
import torch
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
def test_sin(shape):
    check_unary(shape, np.float32, np.sin, ops.sin)


@pytest.mark.parametrize("shape", unary_op_shapes)
def test_cos(shape):
    check_unary(shape, np.float32, np.cos, ops.cos)


@pytest.mark.parametrize("shape", unary_op_shapes)
def test_tan(shape):
    check_unary(shape, np.float32, np.tan, ops.tan)


@pytest.mark.parametrize("shape", unary_op_shapes)
def test_sinh(shape):
    check_unary(shape, np.float32, np.sinh, ops.sinh)


@pytest.mark.parametrize("shape", unary_op_shapes)
def test_cosh(shape):
    check_unary(shape, np.float32, np.cosh, ops.cosh)


@pytest.mark.parametrize("shape", unary_op_shapes)
def test_tanh(shape):
    check_unary(shape, np.float32, np.tanh, ops.tanh)


@pytest.mark.parametrize("shape", unary_op_shapes)
def test_asin(shape):
    check_unary(shape, np.float32, np.arcsin, ops.asin)


@pytest.mark.parametrize("shape", unary_op_shapes)
def test_acos(shape):
    check_unary(shape, np.float32, np.arccos, ops.acos)


@pytest.mark.parametrize("shape", unary_op_shapes)
def test_atan(shape):
    check_unary(shape, np.float32, np.arctan, ops.atan)


@pytest.mark.parametrize("shape", unary_op_shapes)
def test_asinh(shape):
    check_unary(shape, np.float32, np.arcsinh, ops.asinh)


@pytest.mark.parametrize("shape", unary_op_shapes)
def test_acosh(shape):
    check_unary(shape, np.float32, np.arccosh, ops.acosh)


@pytest.mark.parametrize("shape", unary_op_shapes)
def test_atanh(shape):
    check_unary(shape, np.float32, np.arctanh, ops.atanh)


@pytest.mark.parametrize("shape", unary_op_shapes)
def test_exp(shape):
    check_unary(shape, np.float32, np.exp, ops.exp)


@pytest.mark.parametrize("shape", unary_op_shapes)
def test_expm1(shape):
    check_unary(shape, np.float32, np.expm1, ops.expm1)


@pytest.mark.parametrize("shape", unary_op_shapes)
def test_erf(shape):
    check_unary(shape, np.float32, np.vectorize(math.erf), ops.erf)


@pytest.mark.parametrize("shape", unary_op_shapes)
def test_sqrt(shape):
    check_unary(shape, np.float32, np.sqrt, ops.sqrt, positive=True)


@pytest.mark.parametrize("shape", unary_op_shapes)
def test_rsqrt(shape):
    check_unary(shape, np.float32, lambda v: np.reciprocal(np.sqrt(v)), ops.rsqrt, positive=True)


@pytest.mark.parametrize("shape", unary_op_shapes)
def test_log(shape):
    check_unary(shape, np.float32, np.log, ops.log, positive=True)


@pytest.mark.parametrize("shape", unary_op_shapes)
def test_log2(shape):
    check_unary(shape, np.float32, np.log2, ops.log2, positive=True)


@pytest.mark.parametrize("shape", unary_op_shapes)
def test_log10(shape):
    check_unary(shape, np.float32, np.log10, ops.log10, positive=True)


@pytest.mark.parametrize("shape", unary_op_shapes)
def test_log1p(shape):
    check_unary(shape, np.float32, np.log1p, ops.log1p, positive=True)


@pytest.mark.parametrize("shape", unary_op_shapes)
def test_round(shape):
    check_unary(shape, np.float32, np.round, ops.round)


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
def test_floor(a_shape):
    check_unary(a_shape, np.float32, np.floor, ops.floor)


@pytest.mark.parametrize("a_shape", unary_op_shapes)
def test_trunc(a_shape):
    check_unary(a_shape, np.float32, np.trunc, ops.trunc)


@pytest.mark.parametrize("a_shape", unary_op_shapes)
def test_isfinite(a_shape):
    check_unary(a_shape, np.float32, np.isfinite, ops.isfinite)


@pytest.mark.parametrize("a_shape", unary_op_shapes)
def test_isinf(a_shape):
    check_unary(a_shape, np.float32, np.isinf, ops.isinf)


@pytest.mark.parametrize("a_shape", unary_op_shapes)
def test_isnan(a_shape):
    check_unary(a_shape, np.float32, np.isnan, ops.isnan)


@pytest.mark.parametrize("a_shape", [[20]])
def test_cast_from_fp16(a_shape):
    check_unary(a_shape, np.float16, np.int8, lambda x: ops.cast(x, "int8"))
    check_unary(a_shape, np.float16, np.uint8, lambda x: ops.cast(x, "uint8"))

    check_unary(a_shape, np.float16, np.int16, lambda x: ops.cast(x, "int16"))
    check_unary(a_shape, np.float16, np.uint16, lambda x: ops.cast(x, "uint16"))

    check_unary(a_shape, np.float16, np.int32, lambda x: ops.cast(x, "int32"))
    check_unary(a_shape, np.float16, np.uint32, lambda x: ops.cast(x, "uint32"))

    check_unary(a_shape, np.float16, np.int64, lambda x: ops.cast(x, "int64"))
    check_unary(a_shape, np.float16, np.uint64, lambda x: ops.cast(x, "uint64"))


@pytest.mark.parametrize("a_shape", unary_op_shapes)
@pytest.mark.parametrize(
    "a_dtype, b_dtype", [['float16', 'float32'], ['int32', 'float32'], ['int8', 'int32'], ['int32', 'float16']]
)
def test_where(a_shape, a_dtype, b_dtype):
    a = hidet.randn(a_shape, dtype=a_dtype)
    b = hidet.randn(a_shape, dtype=b_dtype)
    c = hidet.ops.where(a > 0.5, a, b)

    c_torch = torch.where(a.torch() > 0.5, a.torch(), b.torch())
    assert str(c.dtype).split('.')[1] == str(c_torch.dtype).split('.')[1]
    np.testing.assert_allclose(c.torch(), c_torch)


if __name__ == '__main__':
    pytest.main([__file__])
