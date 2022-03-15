import argparse
import contextlib
import io
import os
import sys
import time

import git
import numpy as np

from hidet.backend import build
from hidet.baselines.matmul import matmul_cublas, matmul_opt, matmul_cutlass
from hidet.implement import implement, impl_context
from hidet.implement.cuda import CudaGridStaticMatmulImplementer
from hidet.implement.resolve import random_resolve, brute_force_resolve
from hidet.runtime.value import randn, empty, scalar
from hidet.tasks.nn import matmul
from hidet.utils import cuda


def print_latencies(name, latencies, sm_clock=None, mem_clock=None, temperature=None, throttle=None, file=None):
    print('{:>20}: {:.3f} (std {:.3f}) ms [{}] sm_clock {:<4} mem_clock {:<4} temperature {} throttle {}'
          .format(name, np.median(latencies), np.std(latencies), " ".join([f'{v:.3f}' for v in latencies]), sm_clock, mem_clock, temperature, throttle),
          file=file)


def benchmark(warmup=5, number=1, repeat=10, use_brute_force_resolve=True, progress_bar=False, report_dir='./report'):
    workloads = [
        # (80, 2304, 768),
        # (384, 2304, 768)
        (1024, 1024, 1024),
        (2048, 2304, 768),
        *[(16 * T, 2304, 768) for T in [5, 24, 43, 62, 81, 100, 119, 128]]
    ]
    baselines = [
        ('Opt', matmul_opt()),
        ('cutlass', matmul_cutlass()),
        ('cuBLAS', matmul_cublas()),
    ]
    hidet_variants = [
        ('HidetMatmul', (CudaGridStaticMatmulImplementer,)),
    ]
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    short_sha = sha[:7]
    device_name = cuda.query_gpu('name')
    with contextlib.redirect_stdout(io.StringIO()) as f:
        print('{:>25}: {} ({})'.format('Repo commit', short_sha, sha))
        print('{:>25}: {}'.format('GPU', device_name))
        print('{:>25}: {}'.format('Arch', cuda.query_arch()))
        print('{:>25}: {}'.format('SM Clock (MHz)', cuda.query_gpu_current_clock()))
        print('{:>25}: {}'.format('Memory Clock (MHz)', cuda.query_memory_current_clock()))
        print('{:>25}: {}'.format('Compute Capacity', cuda.query_compute_capability()))
        print('{:>25}: {} / {} / {}'.format('Warmup/Number/Repeat', warmup, number, repeat))
        print('{:>25}: {}'.format('Brute-force Resolver', use_brute_force_resolve))
        print()
        for N, M, K in workloads:
            A = randn([N, K], 'float32', 'global', seed=1)
            B = randn([K, M], 'float32', 'global', seed=3)
            C = empty([N, M], 'float32', 'global')
            print("Workload (N x M x K): {} x {} x {}".format(N, M, K))
            print("Workload (N x M x K): {} x {} x {}".format(N, M, K), file=sys.stderr)
            for name, func in baselines:
                time.sleep(1)
                sm_clock, mem_clock, temperature, throttle = cuda.query_gpu_current_clock(), cuda.query_memory_current_clock(), cuda.query_gpu_temperature(), cuda.query_clocks_throttle_reason()
                latencies = func.profile(scalar(N), scalar(M), scalar(K), A, B, C, warmup=warmup, number=number, repeat=repeat)
                print_latencies(name, latencies, sm_clock, mem_clock, temperature, throttle)
                print_latencies(name, latencies, sm_clock, mem_clock, temperature, throttle, sys.stderr)

            for name, allowed in hidet_variants:
                with impl_context(allowed=allowed) as ctx:
                    ir_module = implement(matmul(N, M, K))
                    if use_brute_force_resolve:
                        ir_module = brute_force_resolve(ir_module, warmup=warmup, number=number, repeat=repeat, progress_bar=progress_bar)
                    else:
                        ir_module = random_resolve(ir_module)
                    module = build(ir_module, output_dir=f'./outs/bench/{name}_{N}x{M}x{K}', verbose=False)
                    time.sleep(1)
                    sm_clock, mem_clock, temperature = cuda.query_gpu_current_clock(), cuda.query_memory_current_clock(), cuda.query_gpu_temperature()
                    latencies = module['matmul'].profile(A, B, C, warmup=warmup, number=number, repeat=repeat)
                    print_latencies(name, latencies, sm_clock, mem_clock, temperature)
                    print_latencies(name, latencies, sm_clock, mem_clock, temperature, throttle, sys.stderr)
            print()
            print(file=sys.stderr)
    report = f.getvalue()
    report_name = '{}_{}.report'.format(sha[:7], device_name.replace(' ', '_'))
    os.makedirs(report_dir, exist_ok=True)
    with open(os.path.join(report_dir, report_name), 'w') as f:
        f.write(report)
    print(report)


parser = argparse.ArgumentParser('Hidet benchmark script.')
parser.add_argument('--warmup', type=int, default=5)
parser.add_argument('--number', type=int, default=1)
parser.add_argument('--repeat', type=int, default=10)
parser.add_argument('--resolver', type=str, choices=['random', 'brute'], default='brute')
parser.add_argument('--report_dir', type=str, default='./report')


if __name__ == '__main__':
    args = parser.parse_args()
    with cuda.BenchmarkContext(fix_clock=False):
        benchmark(args.warmup, args.number, args.repeat, args.resolver == 'brute', report_dir=args.report_dir, progress_bar=False)
