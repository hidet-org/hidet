import os
import contextlib

import numpy as np
import io
import git
import argparse

from hidet.backend import build
from hidet.baselines.matmul import matmul_ref, matmul_cublas, matmul_opt, matmul_cutlass
from hidet.implement import implement, impl_context
from hidet.implement.cuda import CudaBlockStaticMatmulSoftPipeImplementer, CudaBlockStaticMatmulNoPipeImplementer, CudaBlockNaiveImplementer, CudaBlockStaticMatmulNoPipeLdgImplementer, CudaBlockStaticMatmulSoftPipeLdgImplementer
from hidet.implement.cuda import CudaGridSplitImplementer, CudaGridNaiveImplementer, CudaWarpTransfer2dImplementer, CudaBlockTransfer2dImplementer, CudaWarpMmaImplementer, CudaWarpFillValueImplementer
from hidet.implement.cuda import CudaThreadNaiveImplementer
from hidet.implement.resolve import random_resolve, brute_force_resolve
from hidet.nn import matmul
from hidet.runtime.value import randn, empty, scalar
from hidet.utils import cuda


def print_latencies(name, latencies):
    print('{:>20}: {:.3f} (std {:.3f}) ms [{}]'.format(name, np.mean(latencies), np.std(latencies), " ".join([f'{v:.3f}' for v in latencies])))


def benchmark(warmup=5, number=1, repeat=10, use_brute_force_resolve=True, progress_bar=False, report_dir='./report'):
    workloads = [
        (1024, 1024, 1024),
        (2048, 2304, 768),
    ]
    baselines = [
        ('Reference', matmul_ref()),
        ('Opt', matmul_opt()),
        ('cutlas', matmul_cutlass()),
        ('cuBLAS', matmul_cublas()),
    ]
    hidet_variants = [
        ('HidetNaive', (CudaGridNaiveImplementer, CudaThreadNaiveImplementer)),
        ('HidetNoPipe', (CudaGridSplitImplementer, CudaBlockStaticMatmulNoPipeImplementer, CudaWarpTransfer2dImplementer, CudaBlockTransfer2dImplementer, CudaWarpMmaImplementer, CudaWarpFillValueImplementer)),
        ('HidetNoPipeLdg', (CudaGridSplitImplementer, CudaBlockStaticMatmulNoPipeLdgImplementer, CudaWarpTransfer2dImplementer, CudaBlockTransfer2dImplementer, CudaWarpMmaImplementer, CudaWarpFillValueImplementer)),
        ('HidetSoftPipe', (CudaGridSplitImplementer, CudaBlockStaticMatmulSoftPipeImplementer, CudaWarpTransfer2dImplementer, CudaBlockTransfer2dImplementer, CudaWarpMmaImplementer, CudaWarpFillValueImplementer)),
        ('HidetSoftPipeLdg', (CudaGridSplitImplementer, CudaBlockStaticMatmulSoftPipeLdgImplementer, CudaWarpTransfer2dImplementer, CudaBlockTransfer2dImplementer, CudaWarpMmaImplementer, CudaWarpFillValueImplementer)),
    ]
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    short_sha = sha[:7]
    device_name = cuda.get_attribute(cuda.Attr.NAME)
    with contextlib.redirect_stdout(io.StringIO()) as f:
        print('Repo commit {} ({})'.format(short_sha, sha))
        print('GPU = {}'.format(device_name))
        print('Arch = {}'.format(cuda.get_attribute(cuda.Attr.ARCH_NAME)))
        print('Compute Capacity = {}'.format(cuda.get_attribute(cuda.Attr.COMPUTE_CAPACITY)))
        print('Warmup/Number/Repeat = {} / {} / {}'.format(warmup, number, repeat))
        print('Use brute-force resolver = {}'.format(use_brute_force_resolve))
        print()
        for N, M, K in workloads:
            A = randn([N, K], 'float32', 'global', seed=1)
            B = randn([K, M], 'float32', 'global', seed=3)
            C = empty([N, M], 'float32', 'global')
            print("Workload (N x M x K): {} x {} x {}".format(N, M, K))
            for name, func in baselines:
                latencies = func.profile(scalar(N), scalar(M), scalar(K), A, B, C, warmup=warmup, number=number, repeat=repeat)
                print_latencies(name, latencies)

            for name, allowed in hidet_variants:
                with impl_context(allowed=allowed) as ctx:
                    ir_module = implement(matmul(N, M, K))
                    if use_brute_force_resolve:
                        ir_module = brute_force_resolve(ir_module, warmup=warmup, number=number, repeat=repeat, progress_bar=progress_bar)
                    else:
                        ir_module = random_resolve(ir_module)
                    module = build(ir_module, output_dir=f'./outs/bench/{name}')
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
    benchmark(args.warmup, args.number, args.repeat, args.resolver == 'brute', args.report_dir)
