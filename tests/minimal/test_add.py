import pytest
import numpy as np
import hidet


def test_add():
    a = hidet.randn([10], device='cuda')
    b = hidet.randn([10], device='cuda')
    c = a + b
    c_np = a.cpu().numpy() + b.cpu().numpy()
    np.testing.assert_allclose(actual=c.cpu().numpy(), desired=c_np, atol=1e-5, rtol=1e-5)


if __name__ == '__main__':
    pytest.main([__file__])
