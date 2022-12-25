import numpy as np
import pytest

import hidet
from hidet import ops
from hidet.testing import check_binary


@pytest.mark.parametrize(
    "a_shape, b_shape, dtype", [[[1, 333, 444], [1, 444, 555], "float32"], [[1, 333, 444], [1, 444, 555], "float16"]]
)
@pytest.mark.parametrize('mma', ['simt', 'wmma', 'mma'])
def test_batch_matmul(a_shape, b_shape, dtype, mma):
    if hidet.cuda.compute_capability() < (8, 0) and mma in ['wmma', 'mma'] and dtype == 'float32':
        pytest.skip('wmma and mma for float32 will triger hidet to use tf32, which is only supported on sm80 and above')
    tolerance = {
        ('float16', 'simt'): 0.5,
        ('float16', 'wmma'): 0.5,
        ('float16', 'mma'): 0.5,
        ('float32', 'simt'): 1e-4,
        ('float32', 'wmma'): 0.05,
        ('float32', 'mma'): 0.05,
    }
    tol = tolerance[(dtype, mma)]
    check_binary(
        a_shape,
        b_shape,
        lambda x, y: np.matmul(x, y),
        lambda x, y: ops.batch_matmul(x, y, mma=mma),
        device='cuda',
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
