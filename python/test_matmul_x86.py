import numpy as np

import hidet
from hidet import ops
from hidet.testing import check_binary

a = hidet.randn([33, 456], dtype='float32', device='cpu')
b = hidet.randn([456, 777], dtype='float32', device='cpu')

c = ops.matmul_x86(a, b)

np.testing.assert_allclose(
    actual=c.numpy(),
    desired=a.numpy() @ b.numpy(),
    rtol=1e-3,
    atol=1e-3
)