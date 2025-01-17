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
from hidet.testing import check_unary, check_unary_dynamic
from hidet.graph.ops.utils import ReduceType
from hidet.graph.ops.reduce import mean, max, prod, min, sum, all, any, argmax, argmin
import hidet


@pytest.mark.parametrize('dtype', [np.float64, np.float32, np.float16])
@pytest.mark.parametrize(
    'shape, dims, keep_dim',
    [[[11, 22, 33], 1, False], [[11, 22, 33], 1, True], [[11, 22, 33], (0, 2), False], [[11, 22, 33], (0, 2), True]],
)
def test_reduce_mean(dtype, shape, dims, keep_dim: bool, device):
    check_unary(
        shape,
        numpy_op=lambda x: np.mean(x, dims, keepdims=keep_dim),
        hidet_op=lambda x: ops.mean(x, dims, keep_dim),
        dtype=dtype,
        atol=1e-5,
        rtol=1e-5,
        device=device,
    )


@pytest.mark.parametrize('dtype', [np.float64, np.float32, np.float16])
@pytest.mark.parametrize(
    'shape, dims, keep_dim',
    [
        [[11, ('s', 22), 33], 1, False],
        [[('x', 11), ('s', 22), 33], 1, True],
        [[11, ('x', 22), ('y', 33)], (0, 2), False],
        [[('a', 11), ('b', 22), ('c', 33)], (0, 2), True],
    ],
)
def test_reduce_mean_dynamic(dtype, shape, dims, keep_dim: bool, device):
    check_unary_dynamic(
        shape,
        numpy_op=lambda x: np.mean(x, dims, keepdims=keep_dim),
        hidet_op=lambda x: ops.mean(x, dims, keep_dim),
        dtype=dtype,
        atol=1e-5,
        rtol=1e-5,
        device=device,
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
def test_var(shape, axis, keep_dim: bool, device):
    check_unary(
        shape,
        numpy_op=lambda x: np.var(x, axis, keepdims=keep_dim),
        hidet_op=lambda x: ops.var(x, axis, keep_dim),
        device=device,
        atol=1e-5,
        rtol=1e-5,
    )


@pytest.mark.parametrize(
    "shape, axis, keep_dim",
    [
        [[1, 24, ('x', 32)], 1, True],
        [[11, ('x', 22), 33], 1, False],
        [[11, ('x', 22), 33], 1, True],
        [[('x', 11), ('y', 22), 33], (0, 2), False],
        [[('x', 11), ('y', 22), 33], (0, 2), True],
    ],
)
def test_var(shape, axis, keep_dim: bool, device):
    check_unary_dynamic(
        shape,
        numpy_op=lambda x: np.var(x, axis, keepdims=keep_dim),
        hidet_op=lambda x: ops.var(x, axis, keep_dim),
        device=device,
        atol=1e-5,
        rtol=1e-5,
    )


@pytest.mark.parametrize("shape", [[11, 22, 34]])
@pytest.mark.parametrize("dims", [1, (0, 2)])
@pytest.mark.parametrize("keep_dim", [False, True])
@pytest.mark.parametrize("reduce_func", [mean, max, prod, min, sum])
def test_reduce_f16(shape, dims, keep_dim: bool, reduce_func, device):
    op_dict = {mean: np.mean, max: np.max, prod: np.prod, min: np.min, sum: np.sum}
    np_op = op_dict[reduce_func]
    check_unary(
        shape,
        numpy_op=lambda x: np_op(x, dims, keepdims=keep_dim),
        hidet_op=lambda x: reduce_func(x, dims, keep_dim),
        device=device,
        dtype=np.float16,
        atol=1e-2,
        rtol=1e-2,
    )


@pytest.mark.parametrize("shape", [[11, 22, 34], [3, 44, 55, 66, 7], [6], [1]])
@pytest.mark.parametrize("dim", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("keep_dim", [False, True])
@pytest.mark.parametrize("reduce_func", [argmax, argmin])
def test_argmax_argmin(shape, dim, keep_dim: bool, reduce_func, device):
    op_dict = {argmax: np.argmax, argmin: np.argmin}
    np_op = op_dict[reduce_func]
    if dim < len(shape):
        check_unary(
            shape,
            numpy_op=lambda x: np_op(x, axis=dim, keepdims=keep_dim),
            hidet_op=lambda x: reduce_func(x, dim=dim, keep_dim=keep_dim),
            device=device,
            dtype=np.float16,
            atol=0,
            rtol=0,
        )


@pytest.mark.parametrize("shape", [[11, 22, 34]])
@pytest.mark.parametrize("axis", [(1,), (0, 2)])
@pytest.mark.parametrize("keep_dim", [False, True])
@pytest.mark.parametrize("reduce_func", [all, any])
def test_reduce_bool(shape, axis, keep_dim: bool, reduce_func, device):
    op_dict = {all: np.all, any: np.any}
    np_op = op_dict[reduce_func]
    check_unary(
        shape,
        numpy_op=lambda x: np_op(x, axis=axis, keepdims=keep_dim),
        hidet_op=lambda x: reduce_func(x, axis=axis, keepdims=keep_dim),
        device=device,
        dtype=bool,
        atol=0,
        rtol=0,
    )


if __name__ == '__main__':
    pytest.main([__file__])
