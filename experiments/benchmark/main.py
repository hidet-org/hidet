from typing import List
import argparse
import datetime
import json
import os
import time

import git
import numpy as np
from tabulate import tabulate

from hidet.backend import build
from hidet.baselines.matmul import matmul_cublas, matmul_opt, matmul_cutlass
from hidet.baselines.conv2d import conv2d_cudnn
from hidet.implement import implement, impl_context
from hidet.implement.cuda import CudaGridStaticMatmulImplementer
from hidet.implement.cuda.conv2d import CudaGridStaticConv2dImplicitGemmImplementer
from hidet.runtime.value import randn, empty, scalar
from hidet.tasks.nn import matmul, conv2d
from hidet.utils import cuda
from hidet.testing import Conv2dSetting


def get_repo_sha(short=False):
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    if short:
        return sha[:7]
    else:
        return sha


def print_latencies(name, latencies):
    print('{:>25}: {:.3f} (std {:.3f}) ms [{}]' .format(name, np.median(latencies), np.std(latencies), " ".join([f'{v:.3f}' for v in latencies])))


def print_enviroment(args):
    with open(os.path.join(args.out_dir, 'env.txt'), 'w') as f:
        f.write(tabulate(
            headers=[
                'Name', 'Value'
            ],
            tabular_data=[
                ['Commit', get_repo_sha()],
                ['GPU', cuda.query_device_name()],
                ['Arch', cuda.query_arch()],
                ['Compute Capacity', cuda.query_compute_capability()],
                ['Lock Clock', args.lock_clock],
                ['Current SM Clock (MHz)', cuda.query_gpu_current_clock()],
                ['Current Memory Clock (MHz)', cuda.query_memory_current_clock()],
                ['Warmup/Number/Repeat', '{} / {} / {}'.format(args.warmup, args.number, args.repeat)]
            ]
        ))


def benchmark_matmul(args):
    workloads = [
        # (1024, 1024, 1024),
        # (5120, 1024, 1024),
        # (1024, 5120, 1024),
        # (1024, 1024, 5120),
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
    names = [name for name, _ in baselines] + [name for name, _ in hidet_variants]
    headers = ['M', 'N', 'K', *names, *[name + '_std' for name in names]]
    table = []
    raw_data = {}
    for M, N, K in workloads:
        workload_key = 'matmul_{}x{}x{}'.format(M, N, K)
        raw_data[workload_key] = {}
        A = randn([M, K], 'float32', 'global', seed=1)
        B = randn([K, N], 'float32', 'global', seed=3)
        C = empty([M, N], 'float32', 'global')
        row: list = [None for _ in range(len(headers))]
        row[:3] = (M, N, K)
        cur = 3
        print(workload_key)
        for name, func in baselines:
            time.sleep(args.cool)
            latencies = func.profile(scalar(M), scalar(N), scalar(K), A, B, C, warmup=args.warmup, number=args.number, repeat=args.repeat)
            print_latencies(name, latencies)
            raw_data[workload_key][name] = latencies
            row[cur] = float(np.median(latencies))
            row[cur + len(names)] = float(np.std(latencies))
            cur += 1

        for name, allowed in hidet_variants:
            with impl_context(allowed=allowed):
                ir_module = implement(matmul(M, N, K))
                module = build(ir_module, output_dir=f'./outs/bench/{name}_{M}x{N}x{K}', verbose=False)
                time.sleep(args.cool)
                latencies = module['matmul'].profile(A, B, C, warmup=args.warmup, number=args.number, repeat=args.repeat)
                print_latencies(name, latencies)
                raw_data[workload_key][name] = latencies
                row[cur] = float(np.median(latencies))
                row[cur + len(names)] = float(np.std(latencies))
                cur += 1
        print()
        table.append(row)
    with open(os.path.join(args.out_dir, 'matmul.raw.json'), 'w') as f:
        json.dump(raw_data, f, indent=2)
    with open(os.path.join(args.out_dir, 'matmul.summary'), 'w') as f:
        f.write(tabulate(table, headers, tablefmt='plain', floatfmt='.3f'))


def benchmark_conv2d(args):
    workloads: List[Conv2dSetting] = list(Conv2dSetting.resnet50_conv2ds(batch_size=1).keys()) + list(Conv2dSetting.resnet50_conv2ds(batch_size=16).keys())
    cudnn_baselines = [
        ('cudnn_implicit_gemm', conv2d_cudnn(algo='implicit_gemm')),
        ('cudnn_auto', conv2d_cudnn(algo='auto'))
    ]
    hidet_variants = [
        ('hidet_implicit_gemm', (CudaGridStaticConv2dImplicitGemmImplementer,))
    ]
    names = [name for name, _ in cudnn_baselines] + [name for name, _ in hidet_variants]
    headers = ['n', 'c', 'h', 'w', 'oc', 'k', 'p', 's', 'gm', 'gn', 'gk'] + names + [name + '_std' for name in names]
    table = []
    raw_data = {}
    for setting in workloads:
        n, ci, hi, wi = setting.batch_size, setting.in_channels, setting.image_size[0], setting.image_size[1]
        co, kx, ky = setting.out_channels, setting.kernel[0], setting.kernel[1]
        ho, wo = setting.output_image_size
        px, py = setting.padding
        sx, sy = setting.stride
        gm = n * ho * wo
        gn = co
        gk = ci * kx * ky
        x = randn([n, ci, hi, wi], 'float32', 'global', seed=1)
        w = randn([co, ci, kx, ky], 'float32', 'global', seed=3)
        y = empty([n, co, ho, wo], 'float32', 'global')
        row = [n, ci, hi, wi, co, f'{kx}x{ky}', f'{px}x{py}', f'{sx}x{sy}', gm, gn, gk]
        raw_data[str(setting)] = {}
        print(str(setting))
        for name, func in cudnn_baselines:
            name = 'cudnn_{}'.format(name)
            time.sleep(args.cool)
            latencies = func.profile(n, ci, hi, wi, co, kx, ky, px, py, sx, sy, x, w, y, warmup=args.warmup, number=args.number, repeat=args.repeat)
            row.append(np.median(latencies))
            print_latencies(name, latencies)
            raw_data[str(setting)][name] = latencies

        for name, allowed in hidet_variants:
            with impl_context(allowed=allowed) as ctx:
                ir_module = implement(conv2d(n, ci, hi, wi, co, (kx, ky), (px, py), (sx, sy)))
                module = build(ir_module, output_dir=f'./outs/bench/{name}_{setting}', keep_ir=False, verbose=False)
                time.sleep(args.cool)
                latencies = module['conv2d'].profile(x, w, y, warmup=args.warmup, number=args.number, repeat=args.repeat)
                row.append(np.median(latencies))
                print_latencies(name, latencies)
                raw_data[str(setting)][name] = latencies
        print()
        table.append(row)
    with open(os.path.join(args.out_dir, 'conv2d.raw.json'), 'w') as f:
        json.dump(raw_data, f, indent=2)
    with open(os.path.join(args.out_dir, 'conv2d.summary'), 'w') as f:
        f.write(tabulate(table, headers, tablefmt='plain', floatfmt='.3f'))


parser = argparse.ArgumentParser('Hidet benchmark script.')
# latency measurement
parser.add_argument('--cool', type=int, default=10)
parser.add_argument('--warmup', type=int, default=5)
parser.add_argument('--number', type=int, default=1)
parser.add_argument('--repeat', type=int, default=10)
parser.add_argument('--no-lock-clock', dest='lock_clock', action='store_false')
# output
parser.add_argument('--out-dir', type=str, default='./results')

if __name__ == '__main__':
    args = parser.parse_args()
    args.out_dir = os.path.join(args.out_dir,
                                '{}_{}'.format(str(datetime.date.today()), get_repo_sha(short=True)),
                                cuda.query_device_name(short=True))
    with cuda.BenchmarkContext(lock_clock=args.lock_clock):
        os.makedirs(args.out_dir, exist_ok=True)
        print_enviroment(args)
        benchmark_matmul(args)
        benchmark_conv2d(args)
