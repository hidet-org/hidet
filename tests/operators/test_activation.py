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
import numpy as np
import pytest
import torch.nn.functional as F

from hidet import ops
from hidet.testing import check_unary, check_binary, check_torch_binary, check_torch_unary


# hidet operators tested against numpy equivalent operators


@pytest.mark.parametrize("shape, dtype", [[[33, 44], "float32"]])
def test_relu_numpy(shape, dtype):
    check_unary(shape, lambda x: np.maximum(x, np.zeros_like(x).astype(dtype)), lambda x: ops.relu(x), dtype=dtype)


@pytest.mark.parametrize("x_shape, slope_shape, dtype", [[[33, 44], [44], "float32"]])
def test_prelu_numpy(x_shape, slope_shape, dtype):
    # without broadcast
    check_binary(
        x_shape,
        x_shape,
        lambda a, b: np.clip(a, 0, np.inf) + np.clip(a, -np.inf, 0) * b,
        lambda a, b: ops.prelu(a, b),
        dtype=dtype,
    )
    # with broadcast
    check_binary(
        x_shape,
        slope_shape,
        lambda a, b: np.clip(a, 0, np.inf) + np.clip(a, -np.inf, 0) * b,
        lambda a, b: ops.prelu(a, b),
        dtype=dtype,
    )


# hidet operators tested against torch equivalent operators


@pytest.mark.parametrize("shape", [[33, 44]])
@pytest.mark.parametrize("dtype", ["float32"])
def test_relu_torch(shape, dtype):
    check_torch_unary(shape, lambda x: F.relu(x), lambda x: ops.relu(x), dtype=dtype, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("x_shape", [[33, 44]])
@pytest.mark.parametrize("slope_shape", [[44]])
@pytest.mark.parametrize("dtype", ["float32"])
def test_prelu_torch(x_shape, slope_shape, dtype):
    check_torch_binary(
        x_shape,
        slope_shape,
        lambda x, y: F.prelu(x, y),
        lambda x, y: ops.prelu(x, y),
        dtype=dtype,
        rtol=1e-5,
        atol=1e-5,
    )


@pytest.mark.parametrize("shape", [[33, 44]])
@pytest.mark.parametrize("dtype", ["float32"])
def test_sigmoid_torch(shape, dtype):
    check_torch_unary(shape, lambda x: F.sigmoid(x), lambda x: ops.sigmoid(x), dtype=dtype, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("shape", [[33, 44]])
@pytest.mark.parametrize("dtype", ["float32"])
def test_hardsigmoid_torch(shape, dtype):
    check_torch_unary(
        shape, lambda x: F.hardsigmoid(x), lambda x: ops.hardsigmoid(x), dtype=dtype, rtol=1e-5, atol=1e-5
    )


@pytest.mark.parametrize("shape", [[33, 44]])
@pytest.mark.parametrize("dtype", ["float32"])
def test_gelu_torch(shape, dtype):
    check_torch_unary(shape, lambda x: F.gelu(x), lambda x: ops.gelu(x), dtype=dtype, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("shape", [[33, 44]])
@pytest.mark.parametrize("dtype", ["float32"])
def test_hardswish_torch(shape, dtype):
    check_torch_unary(shape, lambda x: F.hardswish(x), lambda x: ops.hardswish(x), dtype=dtype, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("shape", [[33, 44]])
@pytest.mark.parametrize("axis", [1])
@pytest.mark.parametrize("dtype", ["float32"])
def test_softmax_torch(shape, axis, dtype):
    check_torch_unary(
        shape, lambda x: F.softmax(x, axis), lambda x: ops.softmax(x, axis), dtype=dtype, rtol=1e-5, atol=1e-5
    )


@pytest.mark.parametrize("shape", [[33, 44]])
@pytest.mark.parametrize("axis", [1])
@pytest.mark.parametrize("dtype", ["float32"])
def test_softmin_torch(shape, axis, dtype):
    check_torch_unary(
        shape, lambda x: F.softmin(x, axis), lambda x: ops.softmin(x, axis), dtype=dtype, rtol=1e-5, atol=1e-5
    )


@pytest.mark.parametrize("shape", [[33, 44]])
@pytest.mark.parametrize("dtype", ["float32"])
def test_logsigmoid_torch(shape, dtype):
    check_torch_unary(shape, lambda x: F.logsigmoid(x), lambda x: ops.logsigmoid(x), dtype=dtype, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("shape", [[33, 44]])
@pytest.mark.parametrize("alpha", [1])
@pytest.mark.parametrize("dtype", ["float32"])
def test_celu_torch(shape, alpha, dtype):
    check_torch_unary(
        shape, lambda x: F.celu(x, alpha), lambda x: ops.celu(x, alpha), dtype=dtype, rtol=1e-5, atol=1e-5
    )


@pytest.mark.parametrize("shape", [[33, 44]])
@pytest.mark.parametrize("lambda_val", [0.5])
@pytest.mark.parametrize("dtype", ["float32"])
def test_hardshrink_torch(shape, lambda_val, dtype):
    check_torch_unary(
        shape,
        lambda x: F.hardshrink(x, lambda_val),
        lambda x: ops.hardshrink(x, lambda_val),
        dtype=dtype,
        rtol=1e-5,
        atol=1e-5,
    )


@pytest.mark.parametrize("shape", [[33, 44]])
@pytest.mark.parametrize("beta", [1])
@pytest.mark.parametrize("threshold", [20])
@pytest.mark.parametrize("dtype", ["float32"])
def test_softplus_torch(shape, beta, threshold, dtype):
    check_torch_unary(
        shape,
        lambda x: F.softplus(x, beta, threshold),
        lambda x: ops.softplus(x, beta, threshold),
        dtype=dtype,
        rtol=1e-5,
        atol=1e-5,
    )


@pytest.mark.parametrize("shape", [[33, 44]])
@pytest.mark.parametrize("dtype", ["float32"])
def test_softsign_torch(shape, dtype):
    check_torch_unary(shape, lambda x: F.softsign(x), lambda x: ops.softsign(x), dtype=dtype, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("shape", [[33, 44]])
@pytest.mark.parametrize("dtype", ["float32"])
def test_tanh_torch(shape, dtype):
    check_torch_unary(shape, lambda x: F.tanh(x), lambda x: ops.tanh(x), dtype=dtype, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("shape", [[33, 44]])
@pytest.mark.parametrize("dtype", ["float32"])
def test_tanhshrink_torch(shape, dtype):
    check_torch_unary(shape, lambda x: F.tanhshrink(x), lambda x: ops.tanhshrink(x), dtype=dtype, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("shape", [[33, 44]])
@pytest.mark.parametrize("min_val", [-1])
@pytest.mark.parametrize("max_val", [1])
@pytest.mark.parametrize("dtype", ["float32"])
def test_hardtanh_torch(shape, min_val, max_val, dtype):
    check_torch_unary(
        shape,
        lambda x: F.hardtanh(x, min_val, max_val),
        lambda x: ops.hardtanh(x, min_val, max_val),
        dtype=dtype,
        rtol=1e-5,
        atol=1e-5,
    )


@pytest.mark.parametrize("shape", [[33, 44]])
@pytest.mark.parametrize("lambda_val", [0.5])
@pytest.mark.parametrize("dtype", ["float32"])
def test_softshrink_torch(shape, lambda_val, dtype):
    check_torch_unary(
        shape,
        lambda x: F.softshrink(x, lambda_val),
        lambda x: ops.softshrink(x, lambda_val),
        dtype=dtype,
        rtol=1e-5,
        atol=1e-5,
    )


if __name__ == '__main__':
    pytest.main([__file__])
