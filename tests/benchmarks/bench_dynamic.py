import argparse
import torch
from hidet.testing.torch_utils import Backend, bench_model


def bench_reduce(backend, mode, dtype, cache):
    comp_backend = Backend(backend, mode, dtype, cache)

    class ReduceModule(torch.nn.Module):
        def forward(self, x):
            return torch.sum(x)

    model = ReduceModule().to(torch.float16).cuda()
    example_inputs = torch.randn((1024), device='cuda', dtype=torch.float16)

    torch._dynamo.mark_dynamic(example_inputs, 0)
    model_op = comp_backend.compile(model)
    model_op(example_inputs)


def bench_matmul(backend, mode, dtype, cache):
    comp_backend = Backend(backend, mode, dtype, cache)
    M = 1024
    N = 1024
    K = 1024

    class MatMulModule(torch.nn.Module):
        def __init__(self):
            super(MatMulModule, self).__init__()
            self.weight = torch.nn.Parameter(torch.randn(K, N))

        def forward(self, x):
            return torch.matmul(x, self.weight)

    model = MatMulModule().to(torch.float16).cuda()
    example_inputs = torch.randn((M, K), device='cuda', dtype=torch.float16)
    torch._dynamo.mark_dynamic(example_inputs, 0)
    model_op = comp_backend.compile(model)
    model_op(example_inputs)


def bench_conv(backend, mode, dtype, cache):
    comp_backend = Backend(backend, mode, dtype, cache)
    N = 2
    C = 3
    H = 224
    W = 224

    class SingleConv(torch.nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
            super(SingleConv, self).__init__()
            self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

        def forward(self, x):
            x = self.conv(x)
            return x

    model = SingleConv(3, 32, (3, 3), stride=(2, 2)).to(torch.float16).cuda()
    example_inputs = torch.randn((N, C, H, W), device='cuda', dtype=torch.float16)
    torch._dynamo.mark_dynamic(example_inputs, 0)
    model_op = comp_backend.compile(model)
    bench_model(model_op, [example_inputs])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Benchmark Dynamic operators')
    parser.add_argument('--dtype', type=str, default='float16', help='Specify precision. E.g., float16')
    parser.add_argument('--backend', type=str, default='hidet', help='torch.compile backend')
    parser.add_argument('--mode', type=str, default='max-autotune', help='torch.compile mode')
    parser.add_argument('--cache', type=str, default='', help='')

    args = parser.parse_args()
    dtype, backend, mode, cache = args.dtype, args.backend, args.mode, args.cache
    bench_conv(backend, mode, dtype, cache)
