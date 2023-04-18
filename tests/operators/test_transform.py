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
from typing import Optional, List
import pytest
import numpy as np
import torch
import hidet as hi
from hidet import ops
from hidet.utils import prod


def check_transform(shape, numpy_op, hidet_op, dtype=np.float32, atol=0, rtol=0):
    # wrap np.array(...) in case shape = []
    data = np.array(np.random.randn(*shape)).astype(dtype)
    numpy_result = numpy_op(data)
    hidet_result = hidet_op(hi.asarray(data).cuda()).cpu().numpy()
    np.testing.assert_allclose(actual=hidet_result, desired=numpy_result, atol=atol, rtol=rtol)


def check_transform_torch(shape, torch_op, hidet_op, dtype=np.float32, atol=0, rtol=0):
    data = torch.asarray(np.array(np.random.randn(*shape)).astype(dtype))
    torch_result = torch_op(data)
    hidet_result = hidet_op(hi.asarray(data).cuda()).cpu().numpy()
    np.testing.assert_allclose(actual=hidet_result, desired=torch_result.cpu().numpy(), atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    "shape, new_shape",
    [
        [[100, 200, 3], [100, 600]],
        [[123, 321], [321, 123]],
        [[123, 321], [-1, 123]],
        [[123, 321], [123 * 321]],
        [[1, 123, 321, 1, 1], [1, 123, 1, 321, 1]],
        [[1], []],
    ],
)
def test_reshape(shape, new_shape):
    check_transform(shape, lambda x: np.reshape(x, new_shape), lambda x: ops.reshape(x, new_shape))


def test_rearrange():
    # Do not test rearrange separately because there is not a corresponding op in numpy,
    # and rearrange has been tested in squeeze, unsqueeze, flatten, transpose because those operators
    # are special cases of rearrange.
    pass


@pytest.mark.parametrize("shape, dims", [[[1, 3, 1, 4], [0, 2]], [[2, 9, 9, 1], [3]], [[1, 1, 1, 1], [0, 1, 2, 3]]])
def test_squeeze(shape, dims):
    check_transform(shape, lambda x: np.squeeze(x, axis=tuple(dims)), lambda x: ops.squeeze(x, dims))


@pytest.mark.parametrize("shape, dims", [[[3, 4], [0, 2]], [[2, 9, 9], [3]], [[], [0, 1, 2, 3]]])
def test_unsqueeze(shape, dims: List[int]):
    check_transform(shape, lambda x: np.expand_dims(x, dims), lambda x: ops.unsqueeze(x, dims))


@pytest.mark.parametrize(
    "shape, start_dim, end_dim",
    [[[33, 44, 55], 0, -1], [[33, 44, 55], 0, 1], [[33, 44, 55], 0, 2], [[33, 44, 55], 1, 2]],
)
def test_flatten(shape, start_dim: int, end_dim: Optional[int]):
    rank = len(shape)
    if start_dim < 0:
        start_dim += rank
    if end_dim < 0:
        end_dim += len(shape)
    flattened_shape = shape[:start_dim] + [prod(shape[start_dim : end_dim + 1])] + shape[end_dim + 1 :]
    check_transform(shape, lambda x: np.reshape(x, flattened_shape), lambda x: ops.flatten(x, start_dim, end_dim))


@pytest.mark.parametrize(
    "shape, axes",
    [[[33, 44, 55], [0, 1, 2]], [[33, 44, 55], [0, 2, 1]], [[33, 44, 55], [2, 1, 0]], [[33, 44, 55], [1, 2, 0]]],
)
def test_transpose(shape, axes):
    check_transform(shape, lambda x: np.transpose(x, axes), lambda x: ops.transpose(x, axes))


@pytest.mark.parametrize(
    "shapes, dtype, axis",
    [
        [[[33, 44, 55], [1, 44, 55], [32, 44, 55]], 'float32', 0],
        [[[33, 1, 55], [33, 8, 55], [33, 111, 55]], 'float32', 1],
        [[[33, 1, 55], [33, 8, 55], [33, 111, 55]], 'float32', -2],
    ],
)
def test_concat(shapes, dtype, axis):
    data_list = [np.random.randn(*shape).astype(dtype) for shape in shapes]
    numpy_result = np.concatenate(data_list, axis)
    hidet_result = ops.concat([hi.asarray(data).cuda() for data in data_list], axis).cpu().numpy()
    np.testing.assert_allclose(actual=hidet_result, desired=numpy_result, rtol=0, atol=0)


@pytest.mark.parametrize("shape, src_type, dst_type", [[[33, 44, 55], "int64", "float32"]])
def test_cast(shape, src_type, dst_type):
    check_transform(shape, lambda x: x.astype(dst_type), lambda x: ops.cast(x, dst_type), dtype=src_type)


@pytest.mark.parametrize("shape, indices_shape, axis", [[[1234, 512], [128], 0], [[12, 34, 56], [2, 2], 1]])
def test_take(shape, indices_shape, axis):
    dim_extent = shape[axis]
    indices = np.random.randint(0, dim_extent - 1, indices_shape).astype(np.int64)
    check_transform(shape, lambda x: np.take(x, indices, axis), lambda x: ops.take(x, hi.asarray(indices).cuda(), axis))


@pytest.mark.parametrize("shape, indices_shape, axis", [[[1234, 512], [2100, 512], 0], [[12, 34, 56], [12, 1, 56], 1]])
def test_gather(shape, indices_shape, axis):
    dim_extent = shape[axis]
    indices = np.random.randint(0, dim_extent - 1, indices_shape).astype(np.int64)
    check_transform_torch(
        shape,
        lambda x: torch.gather(x, axis, torch.asarray(indices)),
        lambda x: ops.gather(x, hi.asarray(indices).cuda(), axis),
    )


@pytest.mark.parametrize(
    "shape, starts, ends, axes, strides",
    [
        [[100, 100, 100], [0, 0, 0], [10, 20, 30], [0, 1, 2], [1, 1, 1]],
        [[100, 100, 100], [5, 6, 7], [10, 20, 30], [0, 1, 2], [1, 1, 1]],
        [[100, 100, 100], [5, 6, 7], [10, 20, 30], [0, 1, 2], [1, 2, 3]],
    ],
)
def test_strided_slice(shape, starts, ends, axes, strides):
    slice_obj = [slice(None, None) for _ in range(len(shape))]
    for start, end, axis, stride in zip(starts, ends, axes, strides):
        slice_obj[axis] = slice(start, end, stride)
    check_transform(shape, lambda x: x[tuple(slice_obj)], lambda x: ops.strided_slice(x, starts, ends, axes, strides))


@pytest.mark.parametrize(
    "shape, broadcast_shape", [[[1, 1, 1], [33, 44, 55]], [[1, 22, 5], [33, 22, 5]], [[1, 55, 1], [33, 55, 44]]]
)
def test_broadcast(shape, broadcast_shape):
    check_transform(shape, lambda x: x + np.zeros(broadcast_shape), lambda x: ops.broadcast(x, broadcast_shape))


if __name__ == '__main__':
    pytest.main([__file__])
