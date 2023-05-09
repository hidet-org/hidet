import pytest
import hidet


@pytest.mark.parametrize('n', [3, 7])
@pytest.mark.parametrize('m', [3, 7])
@pytest.mark.parametrize('k', [-1, 1])
def test_tri(n, m, k):
    import numpy as np

    a = hidet.ops.tri(n, m, k)
    b = np.tri(n, m, k)
    assert np.allclose(a.numpy(), b)
