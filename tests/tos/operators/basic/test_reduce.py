import pytest
import numpy as np
from hidet import ops
from hidet.testing.check import check_unary


@pytest.mark.parametrize(
    'shape, dims, keep_dim',
    [
        [[111, 222, 333], 1, False],
        [[111, 222, 333], 1, True],
        [[111, 222, 333], (0, 2), False],
        [[111, 222, 333], (0, 2), True]
    ]
)
def test_reduce_mean(shape, dims, keep_dim: bool):
    check_unary(shape, numpy_op=lambda x: np.mean(x, dims, keepdims=keep_dim), hidet_op=lambda x: ops.reduce_mean(x, dims, keep_dim), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize(
    "shape, axis, keep_dim",
    [
        [[1, 24, 32], 1, True],
        [[111, 222, 333], 1, False],
        [[111, 222, 333], 1, True],
        [[111, 222, 333], (0, 2), False],
        [[111, 222, 333], (0, 2), True]
    ]
)
def test_var(shape, axis, keep_dim: bool):
    check_unary(shape, numpy_op=lambda x: np.var(x, axis, keepdims=keep_dim), hidet_op=lambda x: ops.reduce_var(x, axis, keep_dim), atol=1e-5, rtol=1e-5)


if __name__ == '__main__':
    pytest.main([__file__])
