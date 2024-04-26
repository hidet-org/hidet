import argparse
import torch
import torchvision
from hidet.testing.torch_utils import bench_torch_model, Backend


def bench_torchvision(model_name, shape, dtype, backend):
    comp_backend = Backend(backend, dtype)

    dtype = getattr(torch, dtype)
    if any(name in model_name for name in ['deeplab', 'fcn', 'lraspp']):
        model_cls = getattr(torchvision.models.segmentation, model_name)
        model = model_cls(weights=None)
        model = model.eval().to(dtype).cuda()
    elif model_name == 'yolov7':
        # TODO: yolov7 don't work right now via pytorch
        model = torch.hub.load(
            'WongKinYiu/yolov7', 'custom', '/tmp/yolov7.pt', autoshape=False, force_reload=True, trust_repo=True
        )
    else:
        model_cls = getattr(torchvision.models, model_name)
        model = model_cls(weights=None)
        model = model.eval().to(dtype).cuda()

    model_inputs = [torch.randn(shape, device='cuda', dtype=dtype)]

    with torch.no_grad(), torch.autocast("cuda"):
        model = comp_backend.compile(model)
        latency = bench_torch_model(model, model_inputs)
        del model
    return latency


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Benchmark Vision Models')
    parser.add_argument('model', type=str, help='Specify model')
    parser.add_argument('--params', type=str, default='1x3x224x224', help='Specify Input Size. E.g., 1x3x224x224')
    parser.add_argument('--dtype', type=str, default='float16', help='Specify precision. E.g., float32')
    parser.add_argument('--backend', type=str, default='hidet', help='torch.compile backend: hidet or max-autotune')
    args = parser.parse_args()

    model, dtype, backend = args.model, args.dtype, args.backend
    shape = [int(d) for d in args.params.split('x')]
    latency = bench_torchvision(model, shape, dtype, backend)
    print(latency)
