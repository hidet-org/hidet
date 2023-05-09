import numpy as np
import pytest

import hidet
from hidet import ops
from hidet.testing import check_binary

for m, k, n in [(1024, 1024, 1024)]:
# for m, k, n in [(333, 444, 555), (1, 123, 3), (13, 17, 381), (423, 432, 233), (1024, 1024, 1024), (373, 367, 311)]:
    a = hidet.randn([m, k], device='cpu')
    b = hidet.randn([k, n], device='cpu')
    c = ops.matmul_x86(a, b)
    np.testing.assert_allclose(
        actual=c.numpy(),
        desired=a.numpy() @ b.numpy(),
        rtol=1e-3,
        atol=1e-3
    )
    hidet_latency = hidet.utils.benchmark_func(
        lambda: ops.matmul_x86(a, b), repeat=30
    )
    print(f'm={m}, n={n}, k={k}: hidet takes {hidet_latency:.2f} ms')


# @pytest.mark.parametrize(
#     "a_shape, b_shape", [[[333, 444], [444, 555]], [[12, 333], [333, 512]]]
# )
# def test_matmul_x86(a_shape, b_shape):
#     check_binary(
#         a_shape,
#         b_shape,
#         lambda x, y: np.matmul(x, y),
#         lambda x, y: ops.matmul_x86(x, y),
#         device='cpu',
#         dtype='float32',
#         atol=1e-3,
#         rtol=1e-3
#     )
#
#
# if __name__ == '__main__':
#     pytest.main([__file__])
