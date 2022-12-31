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
import numpy as np
from hidet import ops
from hidet.testing import check_unary


@pytest.mark.parametrize(
    'shape, dims, keep_dim',
    [[[11, 22, 33], 1, False], [[11, 22, 33], 1, True], [[11, 22, 33], (0, 2), False], [[11, 22, 33], (0, 2), True]],
)
def test_reduce_mean(shape, dims, keep_dim: bool):
    check_unary(
        shape,
        numpy_op=lambda x: np.mean(x, dims, keepdims=keep_dim),
        hidet_op=lambda x: ops.mean(x, dims, keep_dim),
        atol=1e-5,
        rtol=1e-5,
    )


@pytest.mark.parametrize(
    "shape, axis, keep_dim",
    [
        [[1, 24, 32], 1, True],
        [[11, 22, 33], 1, False],
        [[11, 22, 33], 1, True],
        [[11, 22, 33], (0, 2), False],
        [[11, 22, 33], (0, 2), True],
    ],
)
def test_var(shape, axis, keep_dim: bool):
    check_unary(
        shape,
        numpy_op=lambda x: np.var(x, axis, keepdims=keep_dim),
        hidet_op=lambda x: ops.var(x, axis, keep_dim),
        atol=1e-5,
        rtol=1e-5,
    )


if __name__ == '__main__':
    pytest.main([__file__])
