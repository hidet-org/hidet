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


@pytest.mark.parametrize("shape", [[9, 3, 3, 3]])
@pytest.mark.parametrize("num_groups", [3])
@pytest.mark.parametrize("num_channels", [3])
@pytest.mark.parametrize("dtype", ["float32"])
def test_group_norm_torch(shape, num_groups, num_channels, dtype):
    check_torch_unary(
        shape,
        lambda x: F.group_norm(x, num_groups=num_groups),
        lambda x: ops.group_norm(x, num_groups=num_groups, num_channels=num_channels),
        dtype=dtype,
        atol=1e-4,
        rtol=1e-4,
    )


if __name__ == '__main__':
    pytest.main([__file__])
