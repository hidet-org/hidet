from typing import Union
import numpy as np
import pytest

import hidet as hi
from hidet.tos import operators as ops


def check_ternary(a_shape, b_shape, c_shape, numpy_op, hidet_op, dtype: Union[str, np.dtype] = np.float32, atol=0.0, rtol=0.0):
    np.random.seed(1)
    a = np.array(np.random.randn(*a_shape)).astype(dtype)
    b = np.array(np.random.randn(*b_shape)).astype(dtype)
    c = np.array(np.random.randn(*c_shape)).astype(dtype)

    c = np.abs(c)

    print(a)
    print(b)
    print(c)

    numpy_result = numpy_op(a, b, c)
    hidet_args = [hi.array(v).cuda() for v in [a, b, c]]
    hidet_result = hidet_op(*hidet_args).cpu().numpy()
    print(numpy_result)
    print(hidet_result)
    np.testing.assert_allclose(actual=hidet_result, desired=numpy_result, atol=atol, rtol=rtol)


def numpy_batch_norm_2d(data: np.ndarray, running_mean: np.ndarray, running_var: np.ndarray, epsilon: float = 1e-5, axis: int = 1):
    n, c, h, w = data.shape
    assert axis == 1
    assert len(running_mean.shape) == 1 and running_mean.shape[0] == c
    assert len(running_var.shape) == 1 and running_var.shape[0] == c
    running_mean = np.expand_dims(running_mean, axis=(0, 2, 3))
    running_var = np.expand_dims(running_var, axis=(0, 2, 3))
    return (data - running_mean) * np.reciprocal(np.sqrt(running_var + epsilon))


@pytest.mark.parametrize(
    "shape",
    [
        # [1, 200, 20, 20],
        [1, 2, 1, 1],
        [1, 2, 1, 1],
        [1, 2, 1, 1],
        [1, 2, 1, 1],
        # [1, 10, 1, 1],
        # [1, 128, 32, 32],
        # [1, 32, 24, 24],
    ]
)
def test_batch_norm_2d(shape):
    epsilon = 1e-5
    check_ternary(a_shape=shape, b_shape=[shape[1]], c_shape=[shape[1]],
                  numpy_op=lambda a, b, c: numpy_batch_norm_2d(a, b, c, epsilon, 1),
                  hidet_op=lambda a, b, c: ops.batch_norm_infer(a, b, c, epsilon, 1),
                  dtype='float32', atol=1e-5, rtol=1e-5)



