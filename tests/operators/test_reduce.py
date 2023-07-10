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
from hidet.graph.ops.reduce import mean, max, prod


@pytest.mark.parametrize('dtype', [np.float64, np.float32, np.float16])
@pytest.mark.parametrize(
    'shape, dims, keep_dim',
    [[[11, 22, 33], 1, False], [[11, 22, 33], 1, True], [[11, 22, 33], (0, 2), False], [[11, 22, 33], (0, 2), True]],
)
def test_reduce_mean(dtype, shape, dims, keep_dim: bool):
    check_unary(
        shape,
        numpy_op=lambda x: np.mean(x, dims, keepdims=keep_dim),
        hidet_op=lambda x: ops.mean(x, dims, keep_dim),
        dtype=dtype,
        atol=1e-5,
        rtol=1e-5,
        device='all' if dtype != np.float16 else 'cuda',
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
def test_reduce_mean_dynamic(dtype, shape, dims, keep_dim: bool):
    check_unary_dynamic(
        shape,
        numpy_op=lambda x: np.mean(x, dims, keepdims=keep_dim),
        hidet_op=lambda x: ops.mean(x, dims, keep_dim),
        dtype=dtype,
        atol=1e-5,
        rtol=1e-5,
        device='all' if dtype != np.float16 else 'cuda',
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
def test_var(shape, axis, keep_dim: bool):
    check_unary_dynamic(
        shape,
        numpy_op=lambda x: np.var(x, axis, keepdims=keep_dim),
        hidet_op=lambda x: ops.var(x, axis, keep_dim),
        atol=1e-5,
        rtol=1e-5,
    )


@pytest.mark.parametrize("shape", [[11, 22, 34]])
@pytest.mark.parametrize("dims", [1, (0, 2)])
@pytest.mark.parametrize("keep_dim", [False, True])
@pytest.mark.parametrize("reduce_func", [mean, max, prod])
def test_reduce_f16(shape, dims, keep_dim: bool, reduce_func):
    op_dict = {mean: np.mean, max: np.amax, prod: np.prod}
    np_op = op_dict[reduce_type]
    check_unary(
        shape,
        numpy_op=lambda x: np_op(x, dims, keepdims=keep_dim),
        hidet_op=lambda x: reduce_func(x, dims, keep_dim,),
        device='cuda',
        dtype=np.float16,
        atol=1e-2,
        rtol=1e-2,
    )


# TODO: currently dynamic shape with this is not possible
# See reduce_f16.py:reduce_f16 TODO for more details


# @pytest.mark.parametrize("shape", [[11, ('x', 22), ('y', 34)], [('x', 11), 22, ('y', 34)], [('x', 11), ('y', 22), ('z', 34)]])
# @pytest.mark.parametrize("dims", [1, (0, 2)])
# @pytest.mark.parametrize("keep_dim", [False, True])
# @pytest.mark.parametrize("reduce_type", [ReduceType.Average, ReduceType.Max, ReduceType.Product])
# def test_reduce_mean_f16_dynamic(shape, dims, keep_dim: bool, reduce_type):
#     op_dict = {ReduceType.Average: np.mean, ReduceType.Max: np.amax, ReduceType.Product: np.prod}
#     np_op = op_dict[reduce_type]
#     check_unary_dynamic(
#         shape,
#         numpy_op=lambda x: np_op(x, dims, keepdims=keep_dim),
#         hidet_op=lambda x: reduce_f16(x, dims, keep_dim, reduce_type=reduce_type),
#         device='cuda',
#         dtype=np.float16,
#         atol=1e-2,
#         rtol=1e-2,
#     )

if __name__ == '__main__':
    pytest.main([__file__])
