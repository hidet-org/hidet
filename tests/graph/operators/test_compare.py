import numpy as np
import pytest

from hidet import ops
from hidet.testing import check_unary, check_binary


@pytest.mark.parametrize("a_shape, b_shape, dtype", [[[33, 44], [44], "bool"]])
def test_and(a_shape, b_shape, dtype):
    # without broadcast
    check_binary(a_shape, a_shape, lambda a, b: np.logical_and(a, b), lambda a, b: ops.cond_and(a, b), dtype=dtype)
    # with broadcast
    check_binary(a_shape, b_shape, lambda a, b: np.logical_and(a, b), lambda a, b: ops.cond_and(a, b), dtype=dtype)


if __name__ == '__main__':
    pytest.main([__file__])
