from collections import defaultdict, OrderedDict
import time
from tabulate import tabulate
from typing import Tuple
import numpy as np
import nvtx

from hidet.backend import build
from hidet.baselines.pool2d import max_pool2d_cudnn
from hidet.implement import implement, impl_context
from hidet.implement.cuda.pool2d import CudaGridPool2dImplementer
from hidet.runtime.value import TensorValue, randn, empty, scalar, zeros, full, from_list
from hidet.tasks.nn import max_pool2d


def print_latencies(name, latencies):
    print('{:>40}: {:.3f} (std {:.3f}) ms [{}]'.format(name, np.median(latencies), np.std(latencies), " ".join([f'{v:.3f}' for v in latencies])))


def benchmark(warmup=5, number=1, repeat=10, use_nsight_compute=False, keep_ir=True):
    if use_nsight_compute:
        warmup = 0
        number = 1
        repeat = 1
    workloads = [
        (1, 1, 5, 5, 3, 1, 2),
        (1, 64, 112, 112, 3, 1, 2),
        (16, 64, 112, 112, 3, 1, 2),
        (128, 64, 112, 112, 3, 1, 2),
    ]
    cudnn_baselines = [
        ('cudnn_max_pool2d', max_pool2d_cudnn())
    ]
    hidet_variants = [
        ('hidet_pool2d', (CudaGridPool2dImplementer,))
    ]
    print('Repeat = {}'.format(repeat))
    print()
    # table headers
    header = ['n', 'c', 'h', 'w', 'k', 'p', 's', *[name for name, _ in cudnn_baselines], *[name for name, _ in hidet_variants]]
    table = []
    for workload in workloads:
        n, c, h, w, k, p, s = workload
        oh = (h + 2 * p - k) // s + 1
        ow = (w + 2 * p - k) // s + 1
        x = randn([n, c, h, w], 'float32', 'global', seed=1)
        y = zeros([n, c, oh, ow], 'float32', 'global')
        print("Workload {} x {} x {} x {} x {} x {} x {}".format(n, c, h, w, k, p, s))
        row = [n, c, h, w, k, p, s]
        for name, func in cudnn_baselines:
            time.sleep(1)
            latencies = func.profile(n, c, h, w, k, k, p, p, s, s, x, y, warmup=warmup, number=number, repeat=repeat)
            row.append(float(np.median(latencies)))
            print_latencies(name, latencies)
        for name, allowed in hidet_variants:
            with impl_context(allowed=allowed):
                ir_module = implement(max_pool2d(shape=(n, c, h, w), kernel=k, strides=s, padding=p))
                module = build(ir_module, output_dir=f'./outs/bench/{name}_{n}x{c}x{h}x{w}', keep_ir=keep_ir)
                time.sleep(1)
                latencies = module['max_pool2d'].profile(x, y, warmup=warmup, number=number, repeat=repeat)
                row.append(float(np.median(latencies)))
                print_latencies(name, latencies)
        table.append(row)
        print()
    print(tabulate(table, headers=header, floatfmt='.3f', tablefmt='plain', showindex=True))


def verify(keep_ir=True):
    np.set_printoptions(threshold=128 * 128, linewidth=500)
    workloads = [
        (1, 1, 5, 5, 3, 1, 2),
        (1, 64, 112, 112, 3, 1, 2),
    ]
    hidet_variants = [
        ('hidet_pool2d', (CudaGridPool2dImplementer,))
    ]
    for workload in workloads:
        n, c, h, w, k, p, s = workload
        oh = (h + 2 * p - k) // s + 1
        ow = (w + 2 * p - k) // s + 1
        x = randn([n, c, h, w], 'float32', 'global', seed=1)
        print("Workload {} x {} x {} x {} x {} x {} x {}".format(n, c, h, w, k, p, s))

        try:
            ref = max_pool2d_cudnn()
            for name, allowed in hidet_variants:
                with impl_context(allowed=allowed) as ctx:
                    y1 = zeros([n, c, oh, ow], 'float32', 'global')
                    y2 = zeros([n, c, oh, ow], 'float32', 'global')
                    ref(n, c, h, w, k, k, p, p, s, s, x, y1)
                    ir_module = implement(max_pool2d(shape=(n, c, h, w), kernel=k, strides=s, padding=p))
                    module = build(ir_module, output_dir=f'./outs/verify/{name}_{n}x{c}x{h}x{w}x{k}x{p}x{s}', keep_ir=keep_ir)
                    module['max_pool2d'](x, y2)
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
    verify()
    benchmark(use_nsight_compute=False, keep_ir=True)
