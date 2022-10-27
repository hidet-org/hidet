import numpy as np
import pytest

from hidet import ops
from hidet.testing import check_binary


@pytest.mark.parametrize(
    "a_shape, b_shape, dtype", [[[1, 128, 128], [1, 128, 128], "float32"], [[1, 333, 444], [1, 444, 555], "float32"]]
)
@pytest.mark.parametrize('mma', ['simt', 'wmma', 'mma'])
def test_batch_matmul(a_shape, b_shape, dtype, mma):
    mma2tolerance = {'simt': 1e-4, 'wmma': 0.05, 'mma': 0.05}
    tol = mma2tolerance[mma]
    check_binary(
        a_shape,
        b_shape,
        lambda x, y: np.matmul(x, y),
        lambda x, y: ops.batch_matmul(x, y, mma=mma),
        dtype=dtype,
        atol=tol,
        rtol=tol,
    )


@pytest.mark.parametrize(
    "a_shape, b_shape, dtype", [[[1, 128, 128], [128, 128], "float32"], [[333, 444], [444], "float32"]]
)
def test_matmul(a_shape, b_shape, dtype):
    check_binary(
        a_shape, b_shape, lambda x, y: np.matmul(x, y), lambda x, y: ops.matmul(x, y), dtype=dtype, atol=1e-4, rtol=1e-4
    )
