import numpy as np
import pytest

from hidet import ops
from hidet.testing import check_unary


@pytest.mark.parametrize("shape, dtype", [[[33, 44], "float32"]])
def test_relu(shape, dtype):
    check_unary(shape, lambda x: np.maximum(x, np.zeros_like(x).astype(dtype)), lambda x: ops.relu(x), dtype=dtype)


if __name__ == '__main__':
    pytest.main([__file__])
