from collections import defaultdict, OrderedDict
from tabulate import tabulate
from typing import Tuple
import numpy as np
import nvtx

import hidet.tos.operators
import hidet
from hidet.backend import build
from hidet.baselines.matmul import matmul_ref, matmul_cublas
from hidet.baselines.conv2d import conv2d_cudnn, conv2d_reference, conv2d_torch, conv2d_cudnn_available, conv2d_implicit_gemm_reference
from hidet.implement import implement, impl_context
from hidet.implement.resolve import random_resolve, brute_force_resolve
from hidet.implement.cuda.conv2d import CudaGridStaticConv2dImplicitGemmImplementer
from hidet.ir.task import Grid, Host
# from hidet.runtime.value import randn, empty, scalar, zeros, full, from_list
from hidet.utils import cuda, prod
from hidet.transforms import lower
from hidet.tos.tensor import randn, empty, zeros, full


class ConvSetting:
    def __init__(self, batch_size, in_channels, image_size, out_channels, kernel, stride, padding):
        image_size, kernel, stride, padding = self.normalize(image_size, kernel, stride, padding)
        self.batch_size: int = batch_size
        self.in_channels: int = in_channels
        self.image_size: Tuple[int, int] = image_size
        self.out_channels: Tuple[int, int] = out_channels
        self.kernel: Tuple[int, int] = kernel
        self.stride: Tuple[int, int] = stride
        self.padding: Tuple[int, int] = padding
        self.output_image_size = tuple([(image_size[i] + 2 * padding[i] - kernel[i]) // stride[i] + 1 for i in range(2)])

    def __str__(self):
        return 'input_{}x{}x{}x{}__kernel_{}x{}_stride_{}x{}_padding_{}x{}_output_{}x{}x{}x{}_flops_{:.0f}'.format(
            self.batch_size, self.in_channels, *self.image_size, *self.kernel, *self.stride,
            *self.padding, self.batch_size, self.out_channels, *self.output_image_size, self.flops()
        )

    def __repr__(self):
        return str(self)

    def flops(self):
        return self.batch_size * self.out_channels * prod(self.output_image_size) * self.in_channels * prod(self.kernel) / 10 ** 6  # M FLOPs

    @staticmethod
    def normalize(*args):
        for arg in args:
            if not isinstance(arg, (tuple, list)):
                arg = (arg, arg)
            yield arg

    @staticmethod
    def resnet50_conv2ds(batch_size=1):
        workloads = OrderedDict()
        workloads[ConvSetting(batch_size=batch_size, in_channels=3, image_size=224, out_channels=64, kernel=7, stride=2, padding=3)] = 1
        for image_size, channels, repeat in zip([56, 28, 14, 7], [64, 128, 256, 512], [3, 4, 6, 3]):
            if image_size == 56:
                lowering_convs = [
                    (ConvSetting(batch_size=batch_size, in_channels=channels, image_size=image_size, out_channels=channels, kernel=1, stride=1, padding=0), 1),
                    (ConvSetting(batch_size=batch_size, in_channels=channels, image_size=image_size, out_channels=channels, kernel=3, stride=1, padding=1), 1),
                    (ConvSetting(batch_size=batch_size, in_channels=channels, image_size=image_size, out_channels=channels * 4, kernel=1, stride=1, padding=0), 1),
                    (ConvSetting(batch_size=batch_size, in_channels=channels, image_size=image_size, out_channels=channels * 4, kernel=1, stride=1, padding=0), 1)  # skip connection
                ]
            else:
                lowering_convs = [
                    (ConvSetting(batch_size=batch_size, in_channels=channels * 2, image_size=image_size * 2, out_channels=channels, kernel=1, stride=1, padding=0), 1),
                    (ConvSetting(batch_size=batch_size, in_channels=channels, image_size=image_size * 2, out_channels=channels, kernel=3, stride=2, padding=1), 1),
                    (ConvSetting(batch_size=batch_size, in_channels=channels, image_size=image_size, out_channels=channels * 4, kernel=1, stride=1, padding=0), 1),
                    (ConvSetting(batch_size=batch_size, in_channels=channels * 2, image_size=image_size * 2, out_channels=channels * 4, kernel=1, stride=2, padding=0), 1)  # skip connection
                ]
            normal_convs = [
                (ConvSetting(batch_size=batch_size, in_channels=channels * 4, image_size=image_size, out_channels=channels, kernel=1, stride=1, padding=0), repeat - 1),
                (ConvSetting(batch_size=batch_size, in_channels=channels, image_size=image_size, out_channels=channels, kernel=3, stride=1, padding=1), repeat - 1),
                (ConvSetting(batch_size=batch_size, in_channels=channels, image_size=image_size, out_channels=channels * 4, kernel=1, stride=1, padding=0), repeat - 1),
            ]
            for conv, r in lowering_convs + normal_convs:
                if conv not in workloads:
                    workloads[conv] = 0
                workloads[conv] += r
        return workloads

    def __eq__(self, other):
        if len(self.__dict__) != len(other.__dict__):
            return False
        for k in self.__dict__:
            if k not in other.__dict__:
                return False
            if self.__dict__[k] != other.__dict__[k]:
                return False
        return True

    def __hash__(self):
        return hash((self.batch_size, self.in_channels, self.image_size, self.out_channels, self.kernel, self.stride, self.padding))


def print_latencies(name, latencies):
    print('{:>40}: {:.3f} (std {:.3f}) ms [{}]'.format(name, np.median(latencies), np.std(latencies), " ".join([f'{v:.3f}' for v in latencies])))


def benchmark(warmup=10, number=10, repeat=10, use_nsight_compute=False, keep_ir=True, space_level=2):
    if use_nsight_compute:
        warmup = 0
        number = 1
        repeat = 1
    # workloads = list(ConvSetting.resnet50_conv2ds(batch_size=1).keys()) + list(ConvSetting.resnet50_conv2ds(batch_size=16).keys())
    # workloads = [workloads[2]]
    # workloads = ConvSetting.resnet50_conv2ds(batch_size=1)
    workloads = [
        ConvSetting(batch_size=1, in_channels=256, image_size=14, out_channels=256, kernel=3, stride=1, padding=1),
        ConvSetting(batch_size=1, in_channels=512, image_size=7, out_channels=512, kernel=3, stride=1, padding=1)
    ]
    cudnn_baselines = [
        ('fma', 'implicit_gemm'),
        # ('fma', 'implicit_precomp_gemm'),
        # ('fma', 'gemm'),
        # ('fma', 'direct'),
        # ('fma', 'fft'),
        # ('fma', 'fft_tiling'),
        ('fma', 'winograd'),
        # ('fma', 'winograd_nofused'),
        ('fma', 'auto')
    ]
    packed_baselines = [
        # ('reference_implicit_gemm', conv2d_implicit_gemm_reference()),
        # ('reference', conv2d_reference())
    ]
    hidet_variants = [
        ('hidet_implicit_gemm', (CudaGridStaticConv2dImplicitGemmImplementer,))
    ]
    print('Repeat = {}'.format(repeat))
    print()
    # table headers
    header = ['n', 'c', 'h', 'w', 'oc', 'k', 'p', 's', 'gm', 'gn', 'gk']
    for math_mode, algo in cudnn_baselines:
        header.extend([f'cudnn_{algo}'])
    for name, _ in packed_baselines:
        header.append(name)
    for name, _ in hidet_variants:
        header.append(name)
    table = []
    for setting in workloads:
        n, ci, hi, wi = setting.batch_size, setting.in_channels, setting.image_size[0], setting.image_size[1]
        co, kx, ky = setting.out_channels, setting.kernel[0], setting.kernel[1]
        ho, wo = setting.output_image_size
        px, py = setting.padding
        sx, sy = setting.stride
        gm = n * ho * wo
        gn = co
        gk = ci * kx * ky
        x = randn([n, ci, hi, wi], 'float32', device='cuda')
        w = randn([co, ci, kx, ky], 'float32', device='cuda')
        y = empty([n, co, ho, wo], 'float32', device='cuda')
        print("Workload {}".format(setting))

        row = [n, ci, hi, wi, co, f'{kx}x{ky}', f'{px}x{py}', f'{sx}x{sy}', gm, gn, gk]
        for math_mode, algo in cudnn_baselines:
            func, predicate = conv2d_cudnn(math_mode, algo), conv2d_cudnn_available(math_mode, algo)
            name = 'cudnn_{}'.format(algo)
            if predicate(n, ci, hi, wi, co, kx, ky, px, py, sx, sy):
                latencies = func.profile(n, ci, hi, wi, co, kx, ky, px, py, sx, sy, x, w, y, warmup=warmup, number=number, repeat=repeat)
            else:
                latencies = [0.0 for _ in range(repeat)]
            row.append(np.median(latencies))
            print_latencies(name, latencies)

        for name, func in packed_baselines:
            latencies = func.profile(n, ci, hi, wi, co, kx, ky, px, py, sx, sy, x, w, y, warmup=warmup, number=number, repeat=repeat)
            row.append(np.median(latencies))
            print_latencies(name, latencies)

        for name, allowed in hidet_variants:
            with impl_context(allowed=allowed) as ctx:
                yy = hidet.tos.operators.conv2d(hidet.symbol_like(x), w, (px, py), (sx, sy))
                func = hidet.driver.build_task(yy.op.task, space_level=2, opt_level=2, use_cache=True, cache_dir='./outs')
                # module = build(ir_module, output_dir=f'./outs/bench/{name}_{setting}', keep_ir=keep_ir)
                # latencies = module['conv2d'].profile(x, w, y, warmup=warmup, number=number, repeat=repeat)
                latencies = func.profile(x, w, y, warmup=warmup, number=number, repeat=repeat)
                row.append(np.median(latencies))
                print_latencies(name, latencies)
        table.append(row)
        print()
    print(tabulate(table, headers=header, floatfmt='.3f', tablefmt='plain', showindex=True))


def demo_settings():
    settings = ConvSetting.resnet50_conv2ds()
    header = ['n', 'c', 'h', 'w', 'oc', 'k', 'p', 's', 'gemm_m', 'gemm_n', 'gemm_k']
    table = []
    for setting in settings:
        n, ci, hi, wi = setting.batch_size, setting.in_channels, setting.image_size[0], setting.image_size[1]
        co, kx, ky = setting.out_channels, setting.kernel[0], setting.kernel[1]
        ho, wo = setting.output_image_size
        px, py = setting.padding
        sx, sy = setting.stride
        gemm_m = n * ho * wo
        gemm_n = co
        gemm_k = ci * kx * ky
        row = [n, ci, hi, wi, co, f'{kx}x{ky}', f'{px}x{py}', f'{sx}x{sy}', gemm_m, gemm_n, gemm_k]
        table.append(row)
    print(tabulate(table, headers=header))


if __name__ == '__main__':
    # verify()
    benchmark(use_nsight_compute=False, keep_ir=False, space_level=2)
    # test_custom_func()
    # demo_hidet_conv2d()
    # demo_settings()
