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
from hidet import ops
from hidet.testing.utils import check_torch_binary, check_torch_unary, check_torch_binary_with_inputs


binary_op_shapes = [[[1], [200]], [[100, 200], [1, 200]], [[200, 1], [200]]]

unary_op_shapes = [[1], [100], [200]]


@pytest.mark.parametrize("a_shape, b_shape", binary_op_shapes)
def test_add(a_shape, b_shape):
    check_torch_binary(
        a_shape=a_shape,
        b_shape=b_shape,
        torch_func=lambda a, b: a + b,
        hidet_func=lambda a, b: a + b,
        atol=1e-5,
        rtol=1e-5,
    )


@pytest.mark.parametrize("a_shape, b_shape", binary_op_shapes)
def test_sub(a_shape, b_shape):
    check_torch_binary(
        a_shape=a_shape,
        b_shape=b_shape,
        torch_func=lambda a, b: a - b,
        hidet_func=lambda a, b: a - b,
        atol=1e-5,
        rtol=1e-5,
    )


@pytest.mark.parametrize("a_shape, b_shape", binary_op_shapes)
def test_multiply(a_shape, b_shape):
    check_torch_binary(
        a_shape=a_shape,
        b_shape=b_shape,
        torch_func=lambda a, b: a * b,
        hidet_func=lambda a, b: a * b,
        atol=1e-5,
        rtol=1e-5,
    )


@pytest.mark.parametrize("a_shape, b_shape", binary_op_shapes)
def test_divide(a_shape, b_shape):
    check_torch_binary(
        a_shape=a_shape,
        b_shape=b_shape,
        torch_func=lambda a, b: a / b,
        hidet_func=lambda a, b: a / b,
        atol=1e-5,
        rtol=1e-5,
    )


@pytest.mark.parametrize("a_shape, b_shape", binary_op_shapes)
def test_pow(a_shape, b_shape):
    check_torch_binary(
        a_shape=a_shape,
        b_shape=b_shape,
        torch_func=lambda a, b: torch.pow(torch.abs(a), b),
        hidet_func=lambda a, b: ops.pow(ops.abs(a), b),
        atol=1e-5,
        rtol=1e-5,
    )


@pytest.mark.parametrize("shape", unary_op_shapes)
def test_sqrt(shape):
    check_torch_unary(
        shape=shape,
        torch_func=lambda a: torch.sqrt(torch.abs(a)),
        hidet_func=lambda a: ops.sqrt(ops.abs(a)),
        atol=1e-5,
        rtol=1e-5,
    )


@pytest.mark.parametrize("shape", unary_op_shapes)
def test_sin(shape):
    check_torch_unary(
        shape=shape, torch_func=lambda a: torch.sin(a), hidet_func=lambda a: ops.sin(a), atol=1e-5, rtol=1e-5
    )


@pytest.mark.parametrize("shape", unary_op_shapes)
def test_cos(shape):
    check_torch_unary(
        shape=shape, torch_func=lambda a: torch.cos(a), hidet_func=lambda a: ops.cos(a), atol=1e-5, rtol=1e-5
    )


@pytest.mark.parametrize("shape", unary_op_shapes)
def test_tan(shape):
    check_torch_unary(
        shape=shape, torch_func=lambda a: torch.tan(a), hidet_func=lambda a: ops.tan(a), atol=1e-5, rtol=1e-5
    )


@pytest.mark.parametrize("shape", unary_op_shapes)
def test_sinh(shape):
    check_torch_unary(
        shape=shape, torch_func=lambda a: torch.sinh(a), hidet_func=lambda a: ops.sinh(a), atol=1e-5, rtol=1e-5
    )


@pytest.mark.parametrize("shape", unary_op_shapes)
def test_cosh(shape):
    check_torch_unary(
        shape=shape, torch_func=lambda a: torch.cosh(a), hidet_func=lambda a: ops.cosh(a), atol=1e-5, rtol=1e-5
    )


@pytest.mark.parametrize("shape", unary_op_shapes)
def test_tanh(shape):
    check_torch_unary(
        shape=shape, torch_func=lambda a: torch.tanh(a), hidet_func=lambda a: ops.tanh(a), atol=1e-5, rtol=1e-5
    )


@pytest.mark.parametrize("shape", unary_op_shapes)
def test_asin(shape):
    check_torch_unary(
        shape=shape, torch_func=lambda a: torch.asin(a), hidet_func=lambda a: ops.asin(a), atol=1e-5, rtol=1e-5
    )


@pytest.mark.parametrize("shape", unary_op_shapes)
def test_acos(shape):
    check_torch_unary(
        shape=shape, torch_func=lambda a: torch.acos(a), hidet_func=lambda a: ops.acos(a), atol=1e-5, rtol=1e-5
    )


@pytest.mark.parametrize("shape", unary_op_shapes)
def test_atan(shape):
    check_torch_unary(
        shape=shape, torch_func=lambda a: torch.atan(a), hidet_func=lambda a: ops.atan(a), atol=1e-5, rtol=1e-5
    )


@pytest.mark.parametrize("shape", unary_op_shapes)
def test_asinh(shape):
    check_torch_unary(
        shape=shape, torch_func=lambda a: torch.asinh(a), hidet_func=lambda a: ops.asinh(a), atol=1e-5, rtol=1e-5
    )


@pytest.mark.parametrize("shape", unary_op_shapes)
def test_acosh(shape):
    check_torch_unary(
        shape=shape, torch_func=lambda a: torch.acosh(a), hidet_func=lambda a: ops.acosh(a), atol=1e-5, rtol=1e-5
    )


@pytest.mark.parametrize("shape", unary_op_shapes)
def test_atanh(shape):
    check_torch_unary(
        shape=shape, torch_func=lambda a: torch.atanh(a), hidet_func=lambda a: ops.atanh(a), atol=1e-5, rtol=1e-5
    )


@pytest.mark.parametrize("shape", unary_op_shapes)
def test_exp(shape):
    check_torch_unary(
        shape=shape, torch_func=lambda a: torch.exp(a), hidet_func=lambda a: ops.exp(a), atol=1e-5, rtol=1e-5
    )


@pytest.mark.parametrize("shape", unary_op_shapes)
def test_expm1(shape):
    check_torch_unary(
        shape=shape, torch_func=lambda a: torch.expm1(a), hidet_func=lambda a: ops.expm1(a), atol=1e-5, rtol=1e-5
    )


@pytest.mark.parametrize("shape", unary_op_shapes)
def test_erf(shape):
    check_torch_unary(
        shape=shape, torch_func=lambda a: torch.erf(a), hidet_func=lambda a: ops.erf(a), atol=1e-5, rtol=1e-5
    )


@pytest.mark.parametrize("shape", unary_op_shapes)
def test_rsqrt(shape):
    check_torch_unary(
        shape=shape,
        torch_func=lambda a: torch.rsqrt(torch.abs(a)),
        hidet_func=lambda a: ops.rsqrt(ops.abs(a)),
        atol=1e-5,
        rtol=1e-5,
    )


@pytest.mark.parametrize("shape", unary_op_shapes)
def test_log(shape):
    check_torch_unary(
        shape=shape,
        torch_func=lambda a: torch.log(torch.abs(a)),
        hidet_func=lambda a: ops.log(ops.abs(a)),
        atol=1e-5,
        rtol=1e-5,
    )


@pytest.mark.parametrize("shape", unary_op_shapes)
def test_log2(shape):
    check_torch_unary(
        shape=shape,
        torch_func=lambda a: torch.log2(torch.abs(a)),
        hidet_func=lambda a: ops.log2(ops.abs(a)),
        atol=1e-5,
        rtol=1e-5,
    )


@pytest.mark.parametrize("shape", unary_op_shapes)
def test_log10(shape):
    check_torch_unary(
        shape=shape,
        torch_func=lambda a: torch.log10(torch.abs(a)),
        hidet_func=lambda a: ops.log10(ops.abs(a)),
        atol=1e-5,
        rtol=1e-5,
    )


@pytest.mark.parametrize("shape", unary_op_shapes)
def test_log1p(shape):
    check_torch_unary(
        shape=shape,
        torch_func=lambda a: torch.log1p(torch.abs(a)),
        hidet_func=lambda a: ops.log1p(ops.abs(a)),
        atol=1e-5,
        rtol=1e-5,
    )


@pytest.mark.parametrize("shape", unary_op_shapes)
def test_round(shape):
    check_torch_unary(
        shape=shape, torch_func=lambda a: torch.round(a), hidet_func=lambda a: ops.round(a), atol=1e-5, rtol=1e-5
    )


@pytest.mark.parametrize("shape", unary_op_shapes)
def test_neg(shape):
    check_torch_unary(shape=shape, torch_func=lambda a: -a, hidet_func=lambda a: -a, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("shape", unary_op_shapes)
def test_abs(shape):
    check_torch_unary(
        shape=shape, torch_func=lambda a: torch.abs(a), hidet_func=lambda a: ops.abs(a), atol=1e-5, rtol=1e-5
    )


@pytest.mark.parametrize("a_shape, b_shape", binary_op_shapes)
def test_rightshift(a_shape, b_shape):
    a = torch.randint(1, 10000, a_shape)
    b = torch.randint(1, 10, b_shape)
    check_torch_binary_with_inputs(
        a, b, torch_func=lambda a, b: a >> b, hidet_func=lambda a, b: ops.bitwise_right_shift(a, b), atol=0, rtol=0
    )


@pytest.mark.parametrize("a_shape, b_shape", binary_op_shapes)
def test_leftshift(a_shape, b_shape):
    a = torch.randint(1, 10000, a_shape)
    b = torch.randint(1, 10, b_shape)
    check_torch_binary_with_inputs(
        a, b, torch_func=lambda a, b: a << b, hidet_func=lambda a, b: ops.bitwise_left_shift(a, b), atol=0, rtol=0
    )


@pytest.mark.parametrize("a_shape, b_shape", binary_op_shapes)
def test_bitwise_and(a_shape, b_shape):
    check_torch_binary(
        a_shape=a_shape,
        b_shape=b_shape,
        torch_func=lambda a, b: a & b,
        hidet_func=lambda a, b: ops.bitwise_and(a, b),
        dtype='int32',
        atol=0,
        rtol=0,
    )


@pytest.mark.parametrize("a_shape", unary_op_shapes)
def test_bitwise_not(a_shape):
    check_torch_unary(
        shape=a_shape,
        torch_func=lambda a: torch.bitwise_not((a * 10).to(torch.int32)),
        hidet_func=lambda a: ops.bitwise_invert(ops.cast(a * 10, 'int32')),
        atol=0,
        rtol=0,
    )


@pytest.mark.parametrize("a_shape, b_shape", binary_op_shapes)
def test_bitwise_or(a_shape, b_shape):
    check_torch_binary(
        a_shape=a_shape,
        b_shape=b_shape,
        torch_func=lambda a, b: a | b,
        hidet_func=lambda a, b: ops.bitwise_or(a, b),
        dtype='int32',
        atol=0,
        rtol=0,
    )


@pytest.mark.parametrize("a_shape, b_shape", binary_op_shapes)
def test_bitwise_xor(a_shape, b_shape):
    check_torch_binary(
        a_shape=a_shape,
        b_shape=b_shape,
        torch_func=lambda a, b: a ^ b,
        hidet_func=lambda a, b: ops.bitwise_xor(a, b),
        dtype='int32',
        atol=0,
        rtol=0,
    )


@pytest.mark.parametrize("a_shape, b_shape", binary_op_shapes)
def test_minimum(a_shape, b_shape):
    check_torch_binary(
        a_shape=a_shape,
        b_shape=b_shape,
        torch_func=lambda a, b: torch.minimum(a, b),
        hidet_func=lambda a, b: ops.minimum(a, b),
        atol=0,
        rtol=0,
    )


@pytest.mark.parametrize("a_shape, b_shape", binary_op_shapes)
def test_maximum(a_shape, b_shape):
    check_torch_binary(
        a_shape=a_shape,
        b_shape=b_shape,
        torch_func=lambda a, b: torch.maximum(a, b),
        hidet_func=lambda a, b: ops.maximum(a, b),
        atol=0,
        rtol=0,
    )


@pytest.mark.parametrize("a_shape", unary_op_shapes)
def test_ceil(a_shape):
    check_torch_unary(
        shape=a_shape, torch_func=lambda a: torch.ceil(a), hidet_func=lambda a: ops.ceil(a), atol=0, rtol=0
    )


@pytest.mark.parametrize("a_shape", unary_op_shapes)
def test_floor(a_shape):
    check_torch_unary(shape=a_shape, torch_func=lambda a: torch.floor(a), hidet_func=lambda a: ops.floor(a), atol=0)


@pytest.mark.parametrize("a_shape", unary_op_shapes)
def test_trunc(a_shape):
    check_torch_unary(
        shape=a_shape, torch_func=lambda a: torch.trunc(a), hidet_func=lambda a: ops.trunc(a), atol=0, rtol=0
    )


@pytest.mark.parametrize("a_shape", unary_op_shapes)
def test_isfinite(a_shape):
    check_torch_unary(
        shape=a_shape, torch_func=lambda a: torch.isfinite(a), hidet_func=lambda a: ops.isfinite(a), atol=0
    )


@pytest.mark.parametrize("a_shape", unary_op_shapes)
def test_isinf(a_shape):
    check_torch_unary(
        shape=a_shape, torch_func=lambda a: torch.isinf(a), hidet_func=lambda a: ops.isinf(a), atol=0, rtol=0
    )


@pytest.mark.parametrize("a_shape", unary_op_shapes)
def test_isnan(a_shape):
    check_torch_unary(
        shape=a_shape, torch_func=lambda a: torch.isnan(a), hidet_func=lambda a: ops.isnan(a), atol=0, rtol=0
    )


def test_cast_int_subbyte():
    a = torch.randint(low=-8, high=7, size=(4, 4), dtype=torch.int8, device="cuda")
    hidet_a = hidet.from_torch(a)
    torch_b = a.to(torch.float32)
    with hidet.option.context():
        hidet.option.execution_mode("compilation")
        hidet_b = ops.cast(ops.cast(hidet_a, "int4b"), "float32").torch()
    import numpy as np

    np.testing.assert_allclose(actual=hidet_b.cpu().numpy(), desired=torch_b.cpu().numpy(), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("a_shape", unary_op_shapes)
@pytest.mark.parametrize(
    "a_dtype, b_dtype",
    [
        ['float16', 'float32'],
        ['int32', 'float32'],
        ['int8', 'int32'],
        ['int32', 'float16'],
        ['bfloat16', 'float32'],
        ['int32', 'bfloat16'],
    ],
)
def test_where(a_shape, a_dtype, b_dtype, device):
    from hidet.testing import assert_torch_allclose

    a = hidet.randn(a_shape, dtype=a_dtype, device=device)
    b = hidet.randn(a_shape, dtype=b_dtype, device=device)
    c = hidet.ops.where(a > 0.5, a, b)

    c_torch = torch.where(a.torch() > 0.5, a.torch(), b.torch())
    assert str(c.dtype).split('.')[1] == str(c_torch.dtype).split('.')[1]
    assert_torch_allclose(c, c_torch)


if __name__ == '__main__':
    pytest.main([__file__])
