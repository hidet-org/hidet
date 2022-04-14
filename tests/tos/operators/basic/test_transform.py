from typing import Optional
import pytest
import numpy as np
import hidet as hi
import hidet.tos.operators as ops
from hidet.utils import prod


def check_transform(shape, numpy_op, hidet_op, dtype=np.float32):
    # wrap np.array(...) in case shape = []
    data = np.array(np.random.randn(*shape)).astype(dtype)
    numpy_result = numpy_op(data)
    hidet_result = hidet_op(hi.array(data).cuda()).cpu().numpy()
    np.testing.assert_allclose(actual=hidet_result, desired=numpy_result)


@pytest.mark.parametrize(
    "shape, new_shape",
    [[[100, 200, 3], [100, 600]],
     [[123, 321], [321, 123]],
     [[123, 321], [-1, 123]],
     [[123, 321], [123 * 321]],
     [[1], []]]
)
def test_reshape(shape, new_shape):
    check_transform(shape, lambda x: np.reshape(x, new_shape), lambda x: ops.reshape(x, new_shape))

# Do not test rearrange because there is not a corresponding op in numpy,
# and rearrange is used to implement squeeze, unsqueeze, flatten, transpose.


@pytest.mark.parametrize(
    "shape, dims",
    [[[1, 3, 1, 4], [0, 2]],
     [[2, 9, 9, 1], [3]],
     [[1, 1, 1, 1], [0, 1, 2, 3]]]
)
def test_squeeze(shape, dims):
    check_transform(shape, lambda x: np.squeeze(x, axis=tuple(dims)), lambda x: ops.squeeze(x, dims))


@pytest.mark.parametrize(
    "shape, dims",
    [[[3, 4], [0, 2]],
     [[2, 9, 9], [3]],
     [[], [0, 1, 2, 3]]]
)
def test_unsqueeze(shape, dims):
    check_transform(shape, lambda x: np.expand_dims(x, dims), lambda x: ops.unsqueeze(x, dims))


@pytest.mark.parametrize(
    "shape, start_dim, end_dim",
    [
        [[33, 44, 55], 0, None],
        [[33, 44, 55], 0, 1],
        [[33, 44, 55], 0, 2],
        [[33, 44, 55], 1, 2]
    ]
)
def test_flatten(shape, start_dim: int, end_dim: Optional[int]):
    rank = len(shape)
    if start_dim < 0:
        start_dim += rank
    if end_dim is None:
        end_dim = len(shape)
    elif end_dim < 0:
        end_dim += rank
    flattened_shape = shape[:start_dim] + [prod(shape[start_dim:end_dim])] + shape[end_dim:]
    check_transform(shape, lambda x: np.reshape(x, flattened_shape), lambda x: ops.flatten(x, start_dim, end_dim))


@pytest.mark.parametrize(
    "shape, axes",
    [
        [[33, 44, 55], [0, 1, 2]],
        [[33, 44, 55], [0, 2, 1]],
        [[33, 44, 55], [2, 1, 0]],
        [[33, 44, 55], [1, 2, 0]],
    ]
)
def test_transpose(shape, axes):
    check_transform(shape, lambda x: np.transpose(x, axes), lambda x: ops.transpose(x, axes))


def test_concat():
    pass


@pytest.mark.parametrize(
    "shape, src_type, dst_type",
    [
        [[33, 44, 55], "int64", "float32"],
    ]
)
def test_cast(shape, src_type, dst_type):
    check_transform(shape, lambda x: x.astype(dst_type), lambda x: ops.cast(x, dst_type), dtype=src_type)


if __name__ == '__main__':
    pytest.main([__file__])
