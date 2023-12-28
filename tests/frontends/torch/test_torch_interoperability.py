import pytest
import torch
import hidet


def test_as_torch_tensor():
    """
    test __torch_func__ protocol
    """
    a = hidet.randn([32, 32], dtype='float16', device='cuda')
    b = torch.abs(a)
    c = hidet.ops.abs(a)
    torch.testing.assert_close(b, c.torch())


if __name__ == '__main__':
    pytest.main([__file__])
