import pytest
import torch
import torch.nn as nn


@torch.inference_mode
@pytest.mark.requires_cuda
def test_static_shapes():
    model1 = nn.Linear(64, 128, bias=False).cuda().to(dtype=torch.bfloat16)
    model1_opt = torch.compile(model1, backend='hidet', mode='default')

    model2 = nn.Linear(128, 256, bias=False).cuda().to(dtype=torch.bfloat16)
    model2_opt = torch.compile(model2, backend='hidet', mode='default')

    x = torch.randn(64, 64, device="cuda").to(dtype=torch.bfloat16)
    output1 = model1(x)
    output1_opt = model1_opt(x)
    assert torch.allclose(output1, output1_opt, rtol=1e-2, atol=1e-2)

    x = torch.randn(64, 128, device="cuda").to(dtype=torch.bfloat16)
    output2 = model2(x)
    output2_opt = model2_opt(x)
    assert output2.shape == output2_opt.shape
    assert torch.allclose(output2, output2_opt, rtol=1e-2, atol=1e-2)


@torch.inference_mode
@pytest.mark.requires_cuda
@pytest.mark.parametrize("in_size_1", [64])
@pytest.mark.parametrize("in_size_2", [128])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_dynamic_shapes(in_size_1, in_size_2, dtype):
    model1 = nn.Linear(in_size_1, 32, bias=False).cuda().to(dtype)
    model1_opt = torch.compile(model1, backend='hidet', mode='default')

    x = torch.randn(32, in_size_1, device="cuda").to(dtype)
    torch._dynamo.mark_dynamic(x, 0)
    output1 = model1(x)
    output1_opt = model1_opt(x)
    assert torch.allclose(output1, output1_opt, rtol=1e-2, atol=1e-2)

    model2 = nn.Linear(in_size_2, 64, bias=False).cuda().to(dtype)
    model2_opt = torch.compile(model2, backend='hidet', mode='default')

    x = torch.randn(64, in_size_2, device="cuda").to(dtype)
    torch._dynamo.mark_dynamic(x, 0)
    output2 = model2(x)
    output2_opt = model2_opt(x)
    assert output2.shape == output2_opt.shape
    assert torch.allclose(output2, output2_opt, rtol=1e-2, atol=1e-2)


@torch.inference_mode
@pytest.mark.requires_cuda
@pytest.mark.parametrize("dynamo_reset_cache", [True, False])
def test_single_external_weight(dynamo_reset_cache):
    class ModelWithSingleExternalWeight(nn.Module):
        def __init__(self, size=10):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(size, device="cuda"))

        def forward(self, x):
            return x + self.weight

    model = ModelWithSingleExternalWeight().cuda().to(dtype=torch.float32)
    model_opt = torch.compile(model, backend='hidet', mode='default')

    x = torch.randn(10, device="cuda", dtype=torch.float32)

    # First run
    output_ref = model(x)
    output_opt = model_opt(x)

    assert torch.allclose(output_ref, output_opt, rtol=1e-3, atol=1e-3)

    if dynamo_reset_cache:
        torch._dynamo.reset()

    # Second run with a new model instance
    model2 = ModelWithSingleExternalWeight().cuda().to(dtype=torch.float32)
    model2_opt = torch.compile(model2, backend='hidet', mode='default')

    x2 = torch.randn(10, device="cuda", dtype=torch.float32)

    output_ref2 = model2(x2)
    output_opt2 = model2_opt(x2)

    assert torch.allclose(output_ref2, output_opt2, rtol=1e-3, atol=1e-3)


@torch.inference_mode
@pytest.mark.requires_cuda
@pytest.mark.parametrize("dynamo_reset_cache", [True, False])
def test_multiple_use_of_external_weights(dynamo_reset_cache):
    class ModelWithMultipleUseOfTheSameExternalWeight(nn.Module):
        def __init__(self, size=10):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(size, device="cuda"))

        def forward(self, x, y):
            return x + self.weight, y + self.weight

    model = ModelWithMultipleUseOfTheSameExternalWeight().cuda().to(dtype=torch.float32)
    model_opt = torch.compile(model, backend='hidet', mode='default')

    x = torch.randn(10, device="cuda", dtype=torch.float32)
    y = torch.randn(10, device="cuda", dtype=torch.float32)

    # First run
    outputs_ref = model(x, y)
    outputs_opt = model_opt(x, y)

    assert len(outputs_ref) == len(outputs_opt) == 2
    assert torch.allclose(outputs_ref[0], outputs_opt[0], rtol=1e-3, atol=1e-3)
    assert torch.allclose(outputs_ref[1], outputs_opt[1], rtol=1e-3, atol=1e-3)

    if dynamo_reset_cache:
        torch._dynamo.reset()

    # Second run with a new model instance
    model2 = ModelWithMultipleUseOfTheSameExternalWeight().cuda().to(dtype=torch.float32)
    model2_opt = torch.compile(model2, backend='hidet', mode='default')

    x2 = torch.randn(10, device="cuda", dtype=torch.float32)
    y2 = torch.randn(10, device="cuda", dtype=torch.float32)

    outputs_ref2 = model2(x2, y2)
    outputs_opt2 = model2_opt(x2, y2)

    assert torch.allclose(outputs_ref2[0], outputs_opt2[0], rtol=1e-3, atol=1e-3)
    assert torch.allclose(outputs_ref2[1], outputs_opt2[1], rtol=1e-3, atol=1e-3)


@torch.inference_mode
@pytest.mark.requires_cuda
@pytest.mark.parametrize("dynamo_reset_cache", [True, False])
def test_internal_weights(dynamo_reset_cache):
    class BoolComparisonModel(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return x < 42.0

    model = BoolComparisonModel().cuda()
    model_opt = torch.compile(model, backend='hidet', mode='default')

    x = torch.randn(10, device="cuda", dtype=torch.float32) * 100  # Generate values around -100 to 100

    # First run
    output_ref = model(x)
    output_opt = model_opt(x)

    assert output_ref.shape == output_opt.shape
    assert torch.all(output_ref == output_opt)

    if dynamo_reset_cache:
        torch._dynamo.reset()

    # Second run with a new model instance
    model2 = BoolComparisonModel().cuda()
    model2_opt = torch.compile(model2, backend='hidet', mode='default')

    x2 = torch.randn(10, device="cuda", dtype=torch.float32) * 100

    output_ref2 = model2(x2)
    output_opt2 = model2_opt(x2)

    assert output_ref2.shape == output_opt2.shape
    assert torch.all(output_ref2 == output_opt2)


if __name__ == "__main__":
    # For manual testing
    test_static_shapes()
    test_dynamic_shapes(in_size_1=64, in_size_2=128, dtype=torch.bfloat16)
    test_single_external_weight(dynamo_reset_cache=True)
    test_multiple_use_of_external_weights(dynamo_reset_cache=True)
    test_internal_weights(dynamo_reset_cache=True)
