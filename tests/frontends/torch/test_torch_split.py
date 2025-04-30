import pytest
import torch
import torch.nn as nn


# Define constants for split sizes
SPLIT_SIZE_1 = 1024
SPLIT_SIZE_2 = 3 * SPLIT_SIZE_1


class SplitAndMultiplyModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # Split tensor along the last dimension into two parts: 128 and 3*128
        x1, x2 = torch.split(x, [SPLIT_SIZE_1, SPLIT_SIZE_2], dim=-1)
        # Multiply first part by 3, second part by 5
        x1 = x1 * 3
        x2 = x2 * 5
        return x1, x2


@torch.inference_mode
@pytest.mark.requires_cuda
def test_split():
    model = SplitAndMultiplyModel().cuda().to(dtype=torch.bfloat16)
    model_opt = torch.compile(model, backend='hidet', mode='default')

    # First run with batch size 32
    x1 = torch.randn(32, SPLIT_SIZE_1 + SPLIT_SIZE_2, device="cuda", dtype=torch.bfloat16)
    torch._dynamo.mark_dynamic(x1, 0)
    output_ref1 = model(x1)
    output_opt1 = model_opt(x1)

    assert torch.allclose(output_ref1[0], output_opt1[0], rtol=1e-2, atol=1e-2)
    assert torch.allclose(output_ref1[1], output_opt1[1], rtol=1e-2, atol=1e-2)

    # Second run with a different batch size
    x2 = torch.randn(64, SPLIT_SIZE_1 + SPLIT_SIZE_2, device="cuda", dtype=torch.bfloat16)
    torch._dynamo.mark_dynamic(x2, 0)
    output_ref2 = model(x2)
    output_opt2 = model_opt(x2)

    assert torch.allclose(output_ref2[0], output_opt2[0], rtol=1e-2, atol=1e-2)
    assert torch.allclose(output_ref2[1], output_opt2[1], rtol=1e-2, atol=1e-2)


if __name__ == '__main__':
    # For manual testing
    test_split()
