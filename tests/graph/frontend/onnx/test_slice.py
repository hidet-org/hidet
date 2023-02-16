import pytest
import torch

from utils import check_onnx_and_hidet


class SliceModule(torch.nn.Module):
    def __init__(self, indices):
        super().__init__()
        self.indices = indices

    def forward(self, x):
        return x[self.indices]


@pytest.mark.parametrize('shape,indices', [((100,), slice(2, None))])
def test_slice(shape, indices):
    check_onnx_and_hidet(SliceModule(indices), [torch.randn(shape)])
