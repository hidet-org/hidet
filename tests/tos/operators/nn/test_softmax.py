from typing import Union, List, Tuple

import pytest
import numpy as np

import hidet as hi
import hidet.tos.operators as ops

from hidet.testing import check_unary


def numpy_softmax(data, axis):
    data = np.exp(data - np.max(data, axis, keepdims=True))
    data = data / np.sum(data, axis, keepdims=True)
    return data


@pytest.mark.parametrize(
    "shape, axis",
    [
        [[1, 1000], 1],
        [[16, 1000], 1],
        [[1, 1000, 1, 1], 1],
        [[16, 1000, 1, 1], 1],
        [[1, 128, 128, 128], 2]
    ]
)
def test_softmax(shape, axis):
    check_unary(shape, lambda x: numpy_softmax(x, axis), lambda x: ops.softmax(x, axis), dtype='float32', atol=1e-5, rtol=1e-5)
