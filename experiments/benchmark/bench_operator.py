from typing import List
import argparse
import datetime
import json
import os
import time

import git
import numpy as np
from tabulate import tabulate

import hidet
import hidet as hi
from hidet.tos import operators as ops
from hidet.backend import build
from hidet.baselines.matmul import matmul_cublas, matmul_opt, matmul_cutlass
from hidet.baselines.conv2d import conv2d_cudnn
from hidet.implement import implement, impl_context
from hidet.implement.cuda import CudaGridStaticMatmulImplementer
from hidet.implement.cuda.conv2d import CudaGridStaticConv2dImplicitGemmImplementer
from hidet.utils import cuda
from hidet.utils.git_utils import get_repo_commit_date, get_repo_sha
from hidet.testing import Conv2dSetting
from hidet.tos.tensor import randn, empty


def print_latencies(name, latencies):
    print('{:>25}: {:.3f} (std {:.3f}) ms [{}]' .format(name, np.median(latencies), np.std(latencies), " ".join([f'{v:.3f}' for v in latencies])))


def enviroment_info(args) -> str:
    return str(tabulate(
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
        *[(16 * T, 2304, 768) for T in [5, 24, 43, 62, 81, 100, 119, 128]]
    ]
    baselines = []
    hidet_variants = []
    if 'vendor' in args.kernels:
        baselines = [
            ('Opt', matmul_opt()),
            ('cutlass', matmul_cutlass()),
            ('cuBLAS', matmul_cublas()),
        ]
    if 'hidet' in args.kernels:
        hidet_variants = [
            ('HidetMatmul', (CudaGridStaticMatmulImplementer,)),
        ]
    hidet_func = {}
    for idx, (M, N, K) in enumerate(workloads):
        for name, allowed in hidet_variants:
            with impl_context(allowed=allowed):
                a = hi.symbol([M, K], dtype='float32', device='cuda')
                b = hi.symbol([K, N], dtype='float32', device='cuda')
                c = ops.matmul(a, b)
                op = c.trace[0]
                with impl_context(space_level=args.space):
                    ir_module = implement(op.task)
                module = build(ir_module, output_dir=f'./outs/bench/{name}_{M}x{N}x{K}', verbose=False)
                hidet_func[(idx, name)] = module['matmul']
    names = [name for name, _ in baselines] + [name for name, _ in hidet_variants]
    headers = ['M', 'N', 'K', *names, *[name + '_std' for name in names]]
    table = []
    raw_data = {}
    for idx, (M, N, K) in enumerate(workloads):
        workload_key = 'matmul_{}x{}x{}'.format(M, N, K)
        raw_data[workload_key] = {}
        A = randn([M, K], 'float32', device='cuda')
        B = randn([K, N], 'float32', device='cuda')
        C = empty([M, N], 'float32', device='cuda')
        row: list = [None for _ in range(len(headers))]
        row[:3] = (M, N, K)
        cur = 3
        print(workload_key)
        for name, func in baselines:
            time.sleep(args.cool)
            latencies = func.profile(M, N, K, A, B, C, warmup=args.warmup, number=args.number, repeat=args.repeat)
            print_latencies(name, latencies)
            raw_data[workload_key][name] = latencies
            row[cur] = float(np.median(latencies))
            row[cur + len(names)] = float(np.std(latencies))
            cur += 1

        for name, allowed in hidet_variants:
            func = hidet_func[(idx, name)]
            time.sleep(args.cool)
            latencies = func.profile(A, B, C, warmup=args.warmup, number=args.number, repeat=args.repeat)
            print_latencies(name, latencies)
            raw_data[workload_key][name] = latencies
            row[cur] = float(np.median(latencies))
            row[cur + len(names)] = float(np.std(latencies))
            cur += 1
        print()
        table.append(row)
    out_dir = os.path.join(args.out_dir, 'matmul_space{}'.format(args.space))
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'env.txt'), 'w') as f:
        f.write(enviroment_info(args))
    with open(os.path.join(args.out_dir, 'raw.json'), 'w') as f:
        json.dump(raw_data, f, indent=2)
    with open(os.path.join(args.out_dir, 'summary.txt'), 'w') as f:
        f.write(tabulate(table, headers, tablefmt='plain', floatfmt='.3f'))


def benchmark_conv2d(args):
    workloads: List[Conv2dSetting] = list(Conv2dSetting.resnet50_conv2ds(batch_size=1).keys()) + list(Conv2dSetting.resnet50_conv2ds(batch_size=16).keys())
    cudnn_baselines = []
    hidet_variants = []
    if 'vendor' in args.kernels:
        cudnn_baselines = [
            ('cudnn_implicit_gemm', conv2d_cudnn(algo='implicit_gemm')),
            ('cudnn_auto', conv2d_cudnn(algo='auto'))
        ]
    if 'hidet' in args.kernels:
        hidet_variants = [
            ('hidet_implicit_gemm', (CudaGridStaticConv2dImplicitGemmImplementer,))
        ]
    hidet_func = {}
    for idx, setting in enumerate(workloads):
        n, ci, hi, wi = setting.batch_size, setting.in_channels, setting.image_size[0], setting.image_size[1]
        co, kx, ky = setting.out_channels, setting.kernel[0], setting.kernel[1]
        px, py = setting.padding
        sx, sy = setting.stride
        print(setting)
        for name, allowed in hidet_variants:
            with impl_context(allowed=allowed) as ctx:
                x = hidet.symbol([n, ci, hi, wi], dtype='float32', device='cuda')
                w = hidet.symbol([co, ci, kx, ky], dtype='float32', device='cuda')
                y = ops.conv2d(x, w, (px, py), (sx, sy))
                op = y.trace[0]
                with impl_context(space_level=args.space):
                    ir_module = implement(op.task)
                module = build(ir_module, output_dir=f'./outs/bench/{name}_{setting}', keep_ir=False, verbose=False)
                hidet_func[(idx, name)] = module['conv2d']

    names = [name for name, _ in cudnn_baselines] + [name for name, _ in hidet_variants]
    headers = ['n', 'c', 'h', 'w', 'oc', 'k', 'p', 's', 'gm', 'gn', 'gk'] + names + [name + '_std' for name in names]
    table = []
    raw_data = {}
    for idx, setting in enumerate(workloads):
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

        for name, _ in hidet_variants:
            func = hidet_func[(idx, name)]
            latencies = func.profile(x, w, y, warmup=args.warmup, number=args.number, repeat=args.repeat)
            row.append(np.median(latencies))
            print_latencies(name, latencies)
            raw_data[str(setting)][name] = latencies
        print()
        table.append(row)
    out_dir = os.path.join(args.out_dir, 'conv2d_space{}'.format(args.space))
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'env.txt'), 'w') as f:
        f.write(enviroment_info(args))
    with open(os.path.join(out_dir, 'raw.json'), 'w') as f:
        json.dump(raw_data, f, indent=2)
    with open(os.path.join(out_dir, 'summary.txt'), 'w') as f:
        f.write(tabulate(table, headers, tablefmt='plain', floatfmt='.3f'))


parser = argparse.ArgumentParser('Hidet operator benchmark script.')
# latency measurement configs
parser.add_argument('--cool', type=int, default=1)
parser.add_argument('--warmup', type=int, default=5)
parser.add_argument('--number', type=int, default=5)
parser.add_argument('--repeat', type=int, default=5)
parser.add_argument('--no-lock-clock', dest='lock_clock', action='store_false')
parser.add_argument('--workloads', type=str, nargs='+', default=['matmul', 'conv2d'], choices=['matmul', 'conv2d'])
parser.add_argument('--kernels', type=str, nargs='+', default=['vendor', 'hidet'], choices=['vendor', 'hidet'])
# schedule config
parser.add_argument('--space', type=int, choices=[0, 1, 2], default=0)
# output config
parser.add_argument('--out-dir', type=str, default='./results')

if __name__ == '__main__':
    args = parser.parse_args()
    # e.g., './results/2022-03-23_85e5892/V100/operators/'
    args.out_dir = os.path.join(args.out_dir,
                                '{}_{}'.format(get_repo_commit_date(), get_repo_sha(short=True)),
                                cuda.query_device_name(short=True),
                                'operators')
    with cuda.BenchmarkContext(lock_clock=args.lock_clock):
        os.makedirs(args.out_dir, exist_ok=True)
        benchmark_func = {
            'matmul': benchmark_matmul,
            'conv2d': benchmark_conv2d,
        }
        for workload in args.workloads:
            benchmark_func[workload](args)
