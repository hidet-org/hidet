import sys
import os
import argparse
import numpy as np
import torch
import torchvision
import hidet
from bench_utils import enable_compile_server, setup_hidet_flags, bench_torch_model

def bench_torchvision(model_name, shape, dtype):
    setup_hidet_flags(dtype)
    enable_compile_server(True)
    dtype = getattr(torch, dtype)
    if any(name in model_name for name in ['deeplab', 'fcn', 'lraspp']):
        model_cls = getattr(torchvision.models.segmentation, model_name)
    else:
        model_cls = getattr(torchvision.models, model_name)
    model = model_cls(weights=None)
    model = model.eval().to(dtype).cuda()
    torch_inputs = [torch.randn(shape, device='cuda', dtype=dtype)]
    with torch.no_grad(), torch.autocast("cuda"):
        model = torch.compile(model, backend='hidet')
        latency = bench_torch_model(model, torch_inputs)
        del model
    return latency


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Benchmark Vision Models')
    parser.add_argument(
        'model',
        type=str,
        help='Specify model'
    )
    parser.add_argument(
        '--params',
        type=str,
        default='1x3x224x224',
        help='Specify Input Size. E.g., 1x3x224x224'
    )
    parser.add_argument(
        '--dtype',
        type=str,
        default='float16',
        help='Specify precision. E.g., float32'
    )
    args = parser.parse_args()

    model, dtype = args.model, args.dtype
    shape = [int(d) for d in args.params.split('x')]
    latency = bench_torchvision(model, shape, dtype)
    print(latency)