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

from hidet import ops
from hidet.testing import check_unary, check_binary


# hidet operators tested against numpy equivalent operators


@pytest.mark.parametrize("a_shape", [[33, 44]])
@pytest.mark.parametrize("b_shape", [[44]])
@pytest.mark.parametrize("dtype", ["bool"])
def test_logical_and_numpy(a_shape, b_shape, dtype):
    # without broadcast
    check_binary(a_shape, a_shape, lambda a, b: np.logical_and(a, b), lambda a, b: ops.logical_and(a, b), dtype=dtype)
    # with broadcast
    check_binary(a_shape, b_shape, lambda a, b: np.logical_and(a, b), lambda a, b: ops.logical_and(a, b), dtype=dtype)


@pytest.mark.parametrize("a_shape", [[33, 44]])
@pytest.mark.parametrize("b_shape", [[44]])
@pytest.mark.parametrize("dtype", ["bool"])
def test_logical_or_numpy(a_shape, b_shape, dtype):
    # without broadcast
    check_binary(a_shape, a_shape, lambda a, b: np.logical_or(a, b), lambda a, b: ops.logical_or(a, b), dtype=dtype)
    # with broadcast
    check_binary(a_shape, b_shape, lambda a, b: np.logical_or(a, b), lambda a, b: ops.logical_or(a, b), dtype=dtype)


@pytest.mark.parametrize("a_shape", [[33, 44]])
@pytest.mark.parametrize("b_shape", [[44]])
@pytest.mark.parametrize("dtype", ["bool"])
def test_logical_xor_numpy(a_shape, b_shape, dtype):
    # without broadcast
    check_binary(a_shape, a_shape, lambda a, b: np.logical_xor(a, b), lambda a, b: ops.logical_xor(a, b), dtype=dtype)
    # with broadcast
    check_binary(a_shape, b_shape, lambda a, b: np.logical_xor(a, b), lambda a, b: ops.logical_xor(a, b), dtype=dtype)


@pytest.mark.parametrize("a_shape", [[33, 44]])
@pytest.mark.parametrize("dtype", ["bool"])
def test_logical_not_numpy(a_shape, dtype):
    check_unary(a_shape, lambda a: np.logical_not(a), lambda a: ops.logical_not(a), dtype=dtype)


@pytest.mark.parametrize("a_shape", [[33, 44]])
@pytest.mark.parametrize("b_shape", [[44]])
@pytest.mark.parametrize("dtype", ["bool"])
def test_equal_numpy(a_shape, b_shape, dtype):
    # without broadcast
    check_binary(a_shape, a_shape, lambda a, b: np.equal(a, b), lambda a, b: ops.equal(a, b), dtype=dtype)
    # with broadcast
    check_binary(a_shape, b_shape, lambda a, b: np.equal(a, b), lambda a, b: ops.equal(a, b), dtype=dtype)


@pytest.mark.parametrize("a_shape", [[33, 44]])
@pytest.mark.parametrize("b_shape", [[44]])
@pytest.mark.parametrize("dtype", ["bool"])
def test_less_than_equal_numpy(a_shape, b_shape, dtype):
    # without broadcast
    check_binary(a_shape, a_shape, lambda a, b: np.less_equal(a, b), lambda a, b: ops.less_equal(a, b), dtype=dtype)
    # with broadcast
    check_binary(a_shape, b_shape, lambda a, b: np.less_equal(a, b), lambda a, b: ops.less_equal(a, b), dtype=dtype)


@pytest.mark.parametrize("a_shape", [[33, 44]])
@pytest.mark.parametrize("b_shape", [[44]])
@pytest.mark.parametrize("dtype", ["bool"])
def test_less_than_numpy(a_shape, b_shape, dtype):
    # without broadcast
    check_binary(a_shape, a_shape, lambda a, b: np.less(a, b), lambda a, b: ops.less(a, b), dtype=dtype)
    # with broadcast
    check_binary(a_shape, b_shape, lambda a, b: np.less(a, b), lambda a, b: ops.less(a, b), dtype=dtype)


@pytest.mark.parametrize("a_shape", [[33, 44]])
@pytest.mark.parametrize("b_shape", [[44]])
@pytest.mark.parametrize("dtype", ["bool"])
def test_greater_than_equal_numpy(a_shape, b_shape, dtype):
    # without broadcast
    check_binary(
        a_shape, a_shape, lambda a, b: np.greater_equal(a, b), lambda a, b: ops.greater_equal(a, b), dtype=dtype
    )
    # with broadcast
    check_binary(
        a_shape, b_shape, lambda a, b: np.greater_equal(a, b), lambda a, b: ops.greater_equal(a, b), dtype=dtype
    )


@pytest.mark.parametrize("a_shape", [[33, 44]])
@pytest.mark.parametrize("b_shape", [[44]])
@pytest.mark.parametrize("dtype", ["bool"])
def test_greater_than_numpy(a_shape, b_shape, dtype):
    # without broadcast
    check_binary(a_shape, a_shape, lambda a, b: np.greater(a, b), lambda a, b: ops.greater(a, b), dtype=dtype)
    # with broadcast
    check_binary(a_shape, b_shape, lambda a, b: np.greater(a, b), lambda a, b: ops.greater(a, b), dtype=dtype)


if __name__ == '__main__':
    pytest.main([__file__])
