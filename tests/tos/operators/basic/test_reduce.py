import pytest
import numpy as np
import hidet as hi
import hidet.tos.operators as ops


@pytest.mark.parametrize(
    'shape, dims, keep_dim',
    [[[111, 222, 333], 1, False],
     [[111, 222, 333], 1, True],
     [[111, 222, 333], (0, 2), False],
     [[111, 222, 333], (0, 2), True]]
)
def test_reduce_mean(shape, dims, keep_dim: bool):
    dtype = np.float32
    data = np.random.randn(*shape).astype(dtype)
    numpy_result = np.mean(data, axis=dims, keepdims=keep_dim)
    hidet_result = ops.reduce_mean(hi.array(data).cuda(), dims=dims, keep_dim=keep_dim).cpu().numpy()
    np.testing.assert_allclose(actual=hidet_result, desired=numpy_result, atol=1e-5, rtol=1e-5)


if __name__ == '__main__':
    pytest.main([__file__])
