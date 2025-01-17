import pytest
import torch

from hidet.testing.torch_utils import FunctionalModule, check_module


@pytest.mark.parametrize('shape', [(32), (32, 32)])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float16])
def test_empty_like(shape, dtype, device):
    check_module(FunctionalModule(op=torch.empty_like), [torch.randn(shape, dtype=dtype)], device=device)


if __name__ == '__main__':
    pytest.main([__file__])
