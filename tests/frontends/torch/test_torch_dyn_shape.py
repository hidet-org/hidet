import pytest, sys
import torch
import hidet
from hidet.testing.torch_utils import Backend


def no_compilaion(*args, **kwargs):
    assert False, 'At this point must not be compilation, everything should be covered by dynamic shapes'


# REDUCE #
class torch_sum(torch.nn.Module):
    def __init__(self, axis):
        super(torch_sum, self).__init__()
        self.axis = axis

    def forward(self, x):
        return torch.sum(x, dim=self.axis)


def create_model_reduce(axis):
    model = torch_sum(axis=axis)
    return model


@pytest.mark.parametrize('operator', ['reduce'])
@pytest.mark.parametrize('dtype', ['float32'])
@pytest.mark.parametrize('axis', [[1, 2]])
def test_dynamic_shape_w_mark_dynamic(operator, dtype, axis):
    # Testing functionality. No needs in max-autotune
    hidet_backend = Backend(backend='hidet', mode='default', dtype=dtype)
    torch_backend = Backend('eager', None, dtype)
    dtype = getattr(torch, dtype)

    model_creator = getattr(sys.modules[__name__], "create_model_" + operator)
    model = model_creator(axis)
    model = model.eval().to(dtype).cuda()
    with torch.no_grad(), torch.autocast("cuda"):
        hidet_model = hidet_backend.compile(model)
        torch_model = torch_backend.compile(model)

        model_inputs1x = torch.randn(*[2, 16, 16, 3], dtype=dtype, device='cuda')
        # Mark dimension as dynamic
        torch._dynamo.mark_dynamic(model_inputs1x, 0)
        hidet_out = hidet_model(model_inputs1x)
        torch_out = torch_model(model_inputs1x)
        assert torch.allclose(hidet_out, torch_out, rtol=1e-04, atol=1e-04)

        tmp = hidet.drivers.build_task
        hidet.drivers.build_task = no_compilaion

        model_inputs2x = torch.randn(*[3, 16, 16, 3], dtype=dtype, device='cuda')
        hidet_out = hidet_model(model_inputs2x)
        torch_out = torch_model(model_inputs2x)
        assert torch.allclose(hidet_out, torch_out, rtol=1e-04, atol=1e-04)

        model_inputs3x = torch.randn(*[5, 16, 16, 3], dtype=dtype, device='cuda')
        hidet_out = hidet_model(model_inputs3x)
        torch_out = torch_model(model_inputs3x)
        assert torch.allclose(hidet_out, torch_out, rtol=1e-04, atol=1e-04)

        hidet.drivers.build_task = tmp


@pytest.mark.parametrize('operator', ['reduce'])
@pytest.mark.parametrize('dtype', ['float32'])
@pytest.mark.parametrize('axis', [[1, 2]])
def test_dynamic_shape_w_heuristic_mark(operator, dtype, axis):
    # Testing functionality. No needs in max-autotune
    hidet_backend = Backend(backend='hidet', mode='default', dtype=dtype)
    torch_backend = Backend('eager', None, dtype)
    dtype = getattr(torch, dtype)

    model_creator = getattr(sys.modules[__name__], "create_model_" + operator)
    model = model_creator(axis)
    model = model.eval().to(dtype).cuda()
    with torch.no_grad(), torch.autocast("cuda"):
        hidet_model = hidet_backend.compile(model)
        torch_model = torch_backend.compile(model)

        model_inputs1x = torch.randn(*[2, 16, 16, 3], dtype=dtype, device='cuda')
        hidet_out = hidet_model(model_inputs1x)
        torch_out = torch_model(model_inputs1x)
        assert torch.allclose(hidet_out, torch_out, rtol=1e-04, atol=1e-04)

        model_inputs2x = torch.randn(*[3, 16, 16, 3], dtype=dtype, device='cuda')
        hidet_out = hidet_model(model_inputs2x)
        torch_out = torch_model(model_inputs2x)
        assert torch.allclose(hidet_out, torch_out, rtol=1e-04, atol=1e-04)

        tmp = hidet.drivers.build_task
        hidet.drivers.build_task = no_compilaion

        model_inputs3x = torch.randn(*[5, 16, 16, 3], dtype=dtype, device='cuda')
        hidet_out = hidet_model(model_inputs3x)
        torch_out = torch_model(model_inputs3x)
        assert torch.allclose(hidet_out, torch_out, rtol=1e-04, atol=1e-04)

        hidet.drivers.build_task = tmp
