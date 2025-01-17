import torch
import numpy
import pytest


@pytest.mark.parametrize("size", [(32, 32)])
@pytest.mark.parametrize("mode", ["max-autotune", "max-autotune-no-cudagraphs"])
def test_default_stream(size, mode, device):
    if device != 'cuda':
        pytest.skip("Only CUDA device is supported for this test")
    x = torch.rand(size=size, dtype=torch.float32).to(device)
    w = torch.rand(size=size, dtype=torch.float32).to(device)

    def matmul(x):
        return x.matmul(w)

    matmul_opt = torch.compile(matmul, backend='hidet', mode=mode)
    hidet_output = matmul_opt(x)
    torch_output = matmul(x)
    torch_output = torch_output.detach().cpu().numpy()
    hidet_output = hidet_output.detach().cpu().numpy()
    numpy.testing.assert_allclose(torch_output, hidet_output, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("size", [(32, 32)])
@pytest.mark.parametrize("mode", ["max-autotune", "max-autotune-no-cudagraphs"])
def test_new_torch_stream(size, mode, device):
    if device != 'cuda':
        pytest.skip("Only CUDA device is supported for this test")
    x = torch.rand(size=size, dtype=torch.float32).to(device)
    w = torch.rand(size=size, dtype=torch.float32).to(device)
    s = torch.cuda.Stream(device=device)

    def matmul(x):
        return x.matmul(w)

    with torch.cuda.stream(s):
        matmul_opt = torch.compile(matmul, backend='hidet', mode=mode)
        hidet_output = matmul_opt(x)
    torch_output = matmul(x)
    s.synchronize()
    torch_output = torch_output.detach().cpu().numpy()
    hidet_output = hidet_output.detach().cpu().numpy()
    numpy.testing.assert_allclose(torch_output, hidet_output, atol=1e-3, rtol=1e-3)
