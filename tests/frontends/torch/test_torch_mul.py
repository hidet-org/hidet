import pytest
import torch
import torch.nn as nn


# Define constants
SHAPE_SIZE = 512


class MultiplyModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # Multiply input tensor by 3
        return x * 3


@torch.inference_mode
@pytest.mark.requires_cuda
def test_mul():
    model = MultiplyModel().cuda().to(dtype=torch.bfloat16)
    model_opt = torch.compile(model, backend='hidet', mode='default')

    # First run with batch size 32
    x1 = torch.randn(32, SHAPE_SIZE, device="cuda", dtype=torch.bfloat16)
    torch._dynamo.mark_dynamic(x1, 0)
    output_ref1 = model(x1)
    output_opt1 = model_opt(x1)

    assert torch.allclose(output_ref1, output_opt1, rtol=1e-2, atol=1e-2)

    # Second run with a different batch size
    x2 = torch.randn(64, SHAPE_SIZE, device="cuda", dtype=torch.bfloat16)
    torch._dynamo.mark_dynamic(x2, 0)
    output_ref2 = model(x2)
    output_opt2 = model_opt(x2)

    assert torch.allclose(output_ref2, output_opt2, rtol=1e-2, atol=1e-2)


if __name__ == '__main__':
    # For manual testing
    test_mul()
