import argparse
import contextlib
import io
import sys
import os

import git
import numpy as np

from hidet.backend import build
from hidet.baselines.matmul import matmul_ref, matmul_cublas, matmul_opt, matmul_cutlass
from hidet.implement import implement, impl_context
from hidet.implement.cuda import CudaBlockStaticMatmulNoPipeImplementer, CudaBlockNaiveImplementer
from hidet.implement.cuda import CudaGridSplitImplementer, CudaGridNaiveImplementer, CudaWarpTransfer2dImplementer, CudaWarpFillValueImplementer, CudaBlockStaticMatmulSoftPipeLdgWbImplementer
from hidet.implement.cuda import CudaThreadNaiveImplementer, CudaGridStaticMatmulSoftPipePredImplementer
from hidet.implement.resolve import random_resolve, brute_force_resolve
from hidet.runtime.value import randn, empty, scalar
from hidet.tasks.nn import matmul
from hidet.utils import cuda


def print_latencies(name, latencies):
    print('{:>20}: {:.3f} (std {:.3f}) ms [{}]'.format(name, np.mean(latencies), np.std(latencies), " ".join([f'{v:.3f}' for v in latencies])))


def benchmark(warmup=5, number=1, repeat=10, use_brute_force_resolve=True, progress_bar=False, report_dir='./report'):
    workloads = [
        (1024, 1024, 1024),
        (2048, 2304, 768),
        *[(16 * T, 2304, 768) for T in [5, 24, 43, 62, 81, 100, 119, 128]]
    ]
    baselines = [
        ('Reference', matmul_ref()),
        ('Opt', matmul_opt()),
        ('cutlas', matmul_cutlass()),
        ('cuBLAS', matmul_cublas()),
    ]
    hidet_variants = [
        ('HidetNaive', (CudaGridNaiveImplementer, CudaThreadNaiveImplementer)),
        ('HidetNoPipe', (CudaGridSplitImplementer, CudaBlockStaticMatmulNoPipeImplementer, CudaWarpTransfer2dImplementer, CudaBlockNaiveImplementer)),
        ('HidetSoftPipeLdgWb', (CudaGridSplitImplementer, CudaBlockStaticMatmulSoftPipeLdgWbImplementer, CudaWarpTransfer2dImplementer, CudaBlockNaiveImplementer)),
        ('HidetSoftPipePred', (CudaGridSplitImplementer, CudaGridStaticMatmulSoftPipePredImplementer, CudaWarpTransfer2dImplementer, CudaBlockNaiveImplementer)),
    ]
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    short_sha = sha[:7]
    device_name = cuda.get_attribute(cuda.Attr.NAME)
    with contextlib.redirect_stdout(io.StringIO()) as f:
        print('{:>25}: {} ({})'.format('Repo commit', short_sha, sha))
        print('{:>25}: {}'.format('GPU', device_name))
        print('{:>25}: {}'.format('Arch', cuda.get_attribute(cuda.Attr.ARCH_NAME)))
        print('{:>25}: {}'.format('SM Clock (MHz)', cuda.query_gpu_current_clock()))
        print('{:>25}: {}'.format('Memory Clock (MHz)', cuda.query_memory_current_clock()))
        print('{:>25}: {}'.format('Compute Capacity', cuda.get_attribute(cuda.Attr.COMPUTE_CAPACITY)))
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
                latencies = func.profile(scalar(N), scalar(M), scalar(K), A, B, C, warmup=warmup, number=number, repeat=repeat)
                print_latencies(name, latencies)

            for name, allowed in hidet_variants:
                print(name, file=sys.stderr)
                with impl_context(allowed=allowed) as ctx:
                    ir_module = implement(matmul(N, M, K))
                    if use_brute_force_resolve:
                        ir_module = brute_force_resolve(ir_module, warmup=warmup, number=number, repeat=repeat, progress_bar=progress_bar)
                    else:
                        ir_module = random_resolve(ir_module)
                    module = build(ir_module, output_dir=f'./outs/bench/{name}_{N}x{M}x{K}', verbose=False)
                    latencies = module['matmul'].profile(A, B, C, warmup=warmup, number=number, repeat=repeat)
                    print_latencies(name, latencies)
            print()
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
    with cuda.BenchmarkContext(fix_clock=True):
        benchmark(args.warmup, args.number, args.repeat, args.resolver == 'brute', report_dir=args.report_dir, progress_bar=False)
