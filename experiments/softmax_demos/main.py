from collections import defaultdict, OrderedDict
import time
from tabulate import tabulate
from typing import Tuple
import numpy as np
import nvtx

from hidet.backend import build
from hidet.baselines.matmul import matmul_ref, matmul_cublas
from hidet.baselines.conv2d import conv2d_cudnn, conv2d_reference, conv2d_torch, conv2d_cudnn_available, conv2d_implicit_gemm_reference
from hidet.baselines.softmax import softmax_cudnn
from hidet.implement import implement, impl_context
from hidet.implement.resolve import random_resolve, brute_force_resolve
from hidet.implement.cuda.conv2d import CudaGridStaticConv2dImplicitGemmImplementer
from hidet.ir.task import Grid, Host
from hidet.runtime.value import TensorValue, randn, empty, scalar, zeros, full, from_list
from hidet.tasks.nn import matmul, conv2d, softmax
from hidet.utils import cuda, prod
from hidet.transforms import lower


def print_latencies(name, latencies):
    print('{:>40}: {:.3f} (std {:.3f}) ms [{}]'.format(name, np.median(latencies), np.std(latencies), " ".join([f'{v:.3f}' for v in latencies])))


def benchmark(warmup=5, number=1, repeat=10, use_nsight_compute=False, keep_ir=True):
    if use_nsight_compute:
        warmup = 0
        number = 1
        repeat = 1
    workloads = [
        (1, 1000),
        (16, 1000),
    ]
    cudnn_baselines = [
        ('cudnn_softmax', softmax_cudnn())
    ]
    hidet_variants = [
    ]
    print('Repeat = {}'.format(repeat))
    print()
    # table headers
    header = ['m', 'n', *[name for name, _ in cudnn_baselines], *[name for name, _ in hidet_variants]]
    table = []
    for workload in workloads:
        m, n = workload
        x = randn([m, n], 'float32', 'global', seed=1)
        y = randn([m, n], 'float32', 'global', seed=1)
        print("Workload {} x {}".format(m, n))
        row = [m, n]
        for name, func in cudnn_baselines:
            time.sleep(1)
            latencies = func.profile(m, n, x, y, warmup=warmup, number=number, repeat=repeat)
            row.append(float(np.median(latencies)))
            print_latencies(name, latencies)
        for name, allowed in hidet_variants:
            with impl_context(allowed=allowed) as ctx:
                ir_module = implement(softmax(shape=[m, n], axis=1))
                module = build(ir_module, output_dir=f'./outs/bench/{name}_{m}x{n}', keep_ir=keep_ir)
                time.sleep(1)
                latencies = module['matmul'].profile(x, y, warmup=warmup, number=number, repeat=repeat)
                row.append(float(np.median(latencies)))
                print_latencies(name, latencies)
        table.append(row)
        print()
    print(tabulate(table, headers=header, floatfmt='.3f', tablefmt='plain', showindex=True))


def verify(keep_ir=True):
    np.set_printoptions(threshold=128 * 128, linewidth=500)
    workloads = [
        (1, 1000),
        (16, 1000)
    ]
    cudnn_baselines = [
        ('cudnn_softmax', softmax_cudnn())
    ]
    hidet_variants = [
    ]
    for workload in workloads:
        m, n = workload
        x = randn([m, n], 'float32', 'global', seed=1)
        print("Workload {} x {}".format(m, n))

        try:
            ref = softmax_cudnn()
            for name, allowed in hidet_variants:
                with impl_context(allowed=allowed) as ctx:
                    task = softmax(shape=[m, n], axis=1)
                    print(task.compute)
                    continue
                    ir_module = implement(task)
                    module = build(ir_module, output_dir=f'./outs/verify/{name}_{m}x{n}', keep_ir=keep_ir)
                    y1 = zeros([m, n], 'float32', 'global')
                    y2 = zeros([m, n], 'float32', 'global')
                    ref(m, n, x, y1)
                    module['softmax'](x, y2)
                    np.testing.assert_allclose(y1.to_numpy(), y2.to_numpy())

        except AssertionError as e:
            print('x')
            print(x)
            print('y1')
            print(y1)
            print('y2')
            print(y2)
            raise e


if __name__ == '__main__':
    # verify()
    benchmark(use_nsight_compute=False, keep_ir=False)
    # test_custom_func()
    # demo_hidet_conv2d()
    # demo_settings()
