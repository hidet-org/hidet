import argparse
import torch
import torchvision
from hidet.testing.torch_utils import bench_torch_model, Backend
from numpy.testing import assert_allclose


def bench_torchvision(model_name, shape, dtype, backend, mode, cache):
    comp_backend = Backend(backend, mode, dtype, cache)

    dtype = getattr(torch, dtype)
    if any(name in model_name for name in ['deeplab', 'fcn', 'lraspp']):
        model_cls = getattr(torchvision.models.segmentation, model_name)
        model = model_cls(weights=None)
    elif model_name == 'yolov7':
        model = torch.hub.load(
            'WongKinYiu/yolov7', 'custom', '/tmp/yolov7.pt', autoshape=False, force_reload=True, trust_repo=True
        )
    else:
        model_cls = getattr(torchvision.models, model_name)
        model = model_cls(weights=None)
    model = model.eval().to(dtype).cuda()

    model_inputs = [torch.randn(shape, device='cuda', dtype=dtype)]

    eager_outputs = model(*model_inputs)

    with torch.no_grad(), torch.autocast("cuda"):
        model = comp_backend.compile(model)

        latency = bench_torch_model(model, model_inputs)

        compiled_outputs = model(*model_inputs)
        assert len(eager_outputs) == len(compiled_outputs)
        if dtype == torch.float16:
            atol, rtol = 1e-2, 1e-2
            flaky_cases = {'resnet50': 1.5e-1, 'resnext50_32x4d': 5e-2}
            if model_name in flaky_cases:
                atol = flaky_cases[model_name]
        else:
            atol, rtol = 1e-4, 1e-4
        for eager_output, compiled_output in zip(eager_outputs, compiled_outputs):
            assert eager_output.shape == compiled_output.shape
            assert eager_output.dtype == compiled_output.dtype
            assert_allclose(eager_output.cpu().numpy(), compiled_output.cpu().numpy(), atol=atol, rtol=rtol)

        del model
    return latency


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Benchmark Vision Models')
    parser.add_argument('model', type=str, help='Specify model')
    parser.add_argument('--params', type=str, default='1x3x224x224', help='Specify Input Size. E.g., 1x3x224x224')
    parser.add_argument('--dtype', type=str, default='float16', help='Specify precision. E.g., float32')
    parser.add_argument('--backend', type=str, default='hidet', help='torch.compile backend')
    parser.add_argument('--mode', type=str, default='max-autotune', help='torch.compile mode')
    parser.add_argument('--cache', type=str, default='', help='')

    args = parser.parse_args()

    model, dtype, backend, mode, cache = args.model, args.dtype, args.backend, args.mode, args.cache
    shape = [int(d) for d in args.params.split('x')]
    latency = bench_torchvision(model, shape, dtype, backend, mode, cache)

    print(latency)
