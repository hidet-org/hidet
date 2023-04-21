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
from typing import Union
import numpy as np
import pytest
import torch.nn.functional as F

import hidet as hi
from hidet import ops
from hidet.testing import check_unary, check_ternary, check_torch_unary
from hidet.utils import prod


# hidet operators tested against numpy equivalent operators


def numpy_batch_norm_2d(
    data: np.ndarray, running_mean: np.ndarray, running_var: np.ndarray, epsilon: float = 1e-5, axis: int = 1
):
    n, c, h, w = data.shape
    assert axis == 1
    assert len(running_mean.shape) == 1 and running_mean.shape[0] == c
    assert len(running_var.shape) == 1 and running_var.shape[0] == c
    running_mean = np.expand_dims(running_mean, axis=(0, 2, 3))
    running_var = np.expand_dims(running_var, axis=(0, 2, 3))
    return (data - running_mean) * np.reciprocal(np.sqrt(running_var + epsilon))


def numpy_instance_norm(data: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
    dims = tuple(range(2, len(data.shape)))
    mean = data.mean(axis=dims, keepdims=True)
    var = data.var(axis=dims, keepdims=True)
    return (data - mean) / np.sqrt(var + epsilon)


@pytest.mark.parametrize("shape", [[1, 1, 1, 1], [1, 200, 20, 20], [1, 10, 1, 1], [1, 128, 32, 32], [1, 32, 24, 24]])
def test_batch_norm_2d(shape):
    epsilon = 1e-5
    check_ternary(
        a_shape=shape,
        b_shape=[shape[1]],
        c_shape=[shape[1]],
        numpy_op=lambda a, b, c: numpy_batch_norm_2d(a, b, c, epsilon, 1),
        hidet_op=lambda a, b, c: ops.batch_norm_infer(a, b, c, epsilon, 1),
        dtype='float32',
        atol=1e-5,
        rtol=1e-5,
    )


@pytest.mark.parametrize(
    "shape", [[1, 1, 1, 1], [1, 2, 1, 1], [1, 32, 48], [1, 20, 20, 20], [1, 20, 20, 5, 5], [1, 32, 26214]]
)
def test_instance_norm(shape):
    check_unary(shape, numpy_op=numpy_instance_norm, hidet_op=ops.instance_norm, atol=1e-4, rtol=1e-4)


# hidet operators tested against torch equivalent operators


@pytest.mark.parametrize('shape', [[10, 3, 3, 3, 4]])
def test_instance_norm(shape):
    check_torch_unary(shape, lambda x: F.instance_norm(x), lambda x: ops.instance_norm(x), atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize('shape, num_groups', [[[1, 32, 64], 4], [[2, 4, 32], 4], [[1, 4, 32], 1]])
def test_group_norm(shape, num_groups):
    check_torch_unary(
        shape, lambda x: F.group_norm(x, num_groups), lambda x: ops.group_norm(x, num_groups), atol=1e-4, rtol=1e-4
    )


if __name__ == '__main__':
    pytest.main([__file__])
