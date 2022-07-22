import numpy as np
import pytest

from hidet import ops
from hidet.testing import check_binary


@pytest.mark.parametrize(
    "a_shape, b_shape, dtype",
    [
        [[128, 128], [128, 128], "float32"],
        [[512, 1024], [1024, 512], "float32"],
        [[333, 444], [444, 555], "float32"],
    ]
)
def test_matmul(a_shape, b_shape, dtype):
    check_binary(a_shape, b_shape, lambda x, y: np.dot(x, y), lambda x, y: ops.matmul(x, y), dtype, atol=1e-4, rtol=1e-4)
