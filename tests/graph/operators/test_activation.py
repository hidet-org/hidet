import numpy as np
import pytest

from hidet import ops
from hidet.testing import check_unary, check_binary


@pytest.mark.parametrize("shape, dtype", [[[33, 44], "float32"]])
def test_relu(shape, dtype):
    check_unary(shape, lambda x: np.maximum(x, np.zeros_like(x).astype(dtype)), lambda x: ops.relu(x), dtype=dtype)


@pytest.mark.parametrize("x_shape, slope_shape, dtype", [[[33, 44], [44], "float32"]])
def test_prelu(x_shape, slope_shape, dtype):
    # without broadcast
    check_binary(
        x_shape,
        x_shape,
        lambda a, b: np.clip(a, 0, np.inf) + np.clip(a, -np.inf, 0) * b,
        lambda a, b: ops.prelu(a, b),
        dtype=dtype,
    )
    # with broadcast
    check_binary(
        x_shape,
        slope_shape,
        lambda a, b: np.clip(a, 0, np.inf) + np.clip(a, -np.inf, 0) * b,
        lambda a, b: ops.prelu(a, b),
        dtype=dtype,
    )


if __name__ == '__main__':
    pytest.main([__file__])
