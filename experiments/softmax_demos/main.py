from collections import defaultdict, OrderedDict
import time
from tabulate import tabulate
from typing import Tuple
import numpy as np
import nvtx

from hidet.backend import build
from hidet.baselines.softmax import softmax_cudnn
from hidet.implement import implement, impl_context
from hidet.implement.cuda.softmax import CudaGridSoftmaxImplementer
from hidet.runtime.value import TensorValue, randn, empty, scalar, zeros, full, from_list
from hidet.tasks.nn import softmax


def print_latencies(name, latencies):
    print('{:>40}: {:.3f} (std {:.3f}) ms [{}]'.format(name, np.median(latencies), np.std(latencies), " ".join([f'{v:.3f}' for v in latencies])))


def benchmark(warmup=5, number=5, repeat=5, use_nsight_compute=False, keep_ir=True):
    if use_nsight_compute:
        warmup = 0
        number = 1
        repeat = 1
    workloads = [
        (1, 1000, 1, 1),
        (16, 1000, 1, 1),
        *[(bs * 12 * seq_length, seq_length, 1, 1) for bs in [1, 8, 16, 32] for seq_length in [128, 512]]
    ]
    cudnn_baselines = [
        ('cudnn_softmax', softmax_cudnn())
    ]
    hidet_variants = [
        ('hidet_softmax', (CudaGridSoftmaxImplementer,))
    ]
    print('Repeat = {}'.format(repeat))
    print()
    # table headers
    header = ['n', 'c', 'h', 'w', *[name for name, _ in cudnn_baselines], *[name for name, _ in hidet_variants]]
    table = []
    for workload in workloads:
        n, c, h, w = workload
        x = randn([n, c, h, w], 'float32', 'global', seed=1)
        y = randn([n, c, h, w], 'float32', 'global', seed=1)
        print("Workload {} x {} x {} x {}".format(n, c, h, w))
        row = [n, c, h, w]
        for name, func in cudnn_baselines:
            time.sleep(1)
            latencies = func.profile(n, c, h, w, x, y, warmup=warmup, number=number, repeat=repeat)
            row.append(float(np.median(latencies)))
            print_latencies(name, latencies)
        for name, allowed in hidet_variants:
            with impl_context(allowed=allowed):
                ir_module = implement(softmax(shape=[n, c, h, w], axis=1))
                module = build(ir_module, output_dir=f'./outs/bench/{name}_{n}x{c}x{h}x{w}', keep_ir=keep_ir)
                time.sleep(1)
                latencies = module['softmax'].profile(x, y, warmup=warmup, number=number, repeat=repeat)
                row.append(float(np.median(latencies)))
                print_latencies(name, latencies)
        table.append(row)
        print()
    print(tabulate(table, headers=header, floatfmt='.3f', tablefmt='plain', showindex=True))


def verify(keep_ir=True):
    np.set_printoptions(threshold=128 * 128, linewidth=500)
    workloads = [
        (1, 10, 1, 1),
        (1, 1000, 1, 1),
        (16, 1000, 1, 1),
        (16, 1000, 16, 16),
        (98304, 512, 1, 1),
    ]
    cudnn_baselines = [
        ('cudnn_softmax', softmax_cudnn())
    ]
    hidet_variants = [
        ('hidet_softmax', (CudaGridSoftmaxImplementer,))
    ]
    for workload in workloads:
        n, c, h, w = workload
        x = randn([n, c, h, w], 'float32', 'global', seed=1)
        print("Workload {} x {} x {} x {}".format(n, c, h, w))

        try:
            ref = softmax_cudnn()
            for name, allowed in hidet_variants:
                with impl_context(allowed=allowed) as ctx:
                    task = softmax(shape=[n, c, h, w], axis=1)
                    ir_module = implement(task)
                    module = build(ir_module, output_dir=f'./outs/verify/{name}_{n}x{c}x{h}x{w}', keep_ir=keep_ir)
                    y1 = zeros([n, c, h, w], 'float32', 'global')
                    y2 = zeros([n, c, h, w], 'float32', 'global')
                    ref(n, c, h, w, x, y1)
                    module['softmax'](x, y2)
                    np.testing.assert_allclose(y1.to_numpy(), y2.to_numpy(), atol=1e-5, rtol=1e-5)

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
    benchmark(use_nsight_compute=False, keep_ir=True)
