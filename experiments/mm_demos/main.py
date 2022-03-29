import os
import time

import numpy as np

from hidet.backend import build
from hidet.baselines.matmul import matmul_cublas, matmul_opt, matmul_cutlass
from hidet.implement import implement, impl_context
from hidet.implement.cuda import CudaGridStaticMatmulImplementer
from hidet.implement.resolve import random_resolve, brute_force_resolve
from hidet.ir.task import Grid, Host
from hidet.tasks.nn import matmul
from hidet.utils import cuda
from hidet.tos.tensor import from_numpy
from hidet.tos.tensor import randn, empty, zeros, full


def print_latencies(name, latencies, file=None):
    print('{:>20}: {:.3f} (std {:.3f}) ms [{}]'.format(name, np.median(latencies), np.std(latencies), " ".join([f'{v:.3f}' for v in latencies])), file=file)


def benchmark(warmup=5, number=1, repeat=10, use_brute_force_resolve=False, progress_bar=True, use_nsight_compute=False, keep_ir=False):
    if use_nsight_compute:
        warmup = 0
        number = 1
        repeat = 1
    workloads = [
        # (222, 333, 444),
        (1024, 1024, 1024),
        (1024 + 22, 1024 + 33, 1024 + 44),
        *[(16 * T, 2304, 768) for T in [5, 24, 43, 62, 81, 100, 119, 128]],
        # *[(16 * T, 2304, 768) for T in [5, 81, 128]]
    ]
    baselines = [
        ('Opt', matmul_opt()),
        ('cutlas', matmul_cutlass()),
        ('cublas', matmul_cublas()),
    ]
    hidet_variants = [
        # ('HidetNaive', (CudaGridNaiveImplementer, CudaThreadNaiveImplementer)),
        # ('HidetNoPipe', (CudaGridSplitImplementer, CudaBlockStaticMatmulNoPipeImplementer, CudaBlockNaiveImplementer)),
        # ('HidetSoftPipeLdgWb', (CudaGridSplitImplementer, CudaBlockStaticMatmulSoftPipeLdgWbImplementer, CudaBlockNaiveImplementer)),
        # ('HidetSoftPipePred', (CudaGridStaticMatmulSoftPipePredImplementer, CudaBlockNaiveImplementer)),
        ('HidetMatmul', (CudaGridStaticMatmulImplementer,))
    ]
    print('Repeat = {}'.format(repeat))
    print('Brute-force resolver = {}'.format(use_brute_force_resolve))
    print()
    os.makedirs('./outs/bench', exist_ok=True)
    with open('./outs/bench/summary.txt', 'w') as f:
        for N, M, K in workloads:
            A = randn([N, K], 'float32', device='cuda')
            B = randn([K, M], 'float32', device='cuda')
            C = empty([N, M], 'float32', device='cuda')
            print("Workload (N x M x K): {} x {} x {}".format(N, M, K))
            print("Workload (N x M x K): {} x {} x {}".format(N, M, K), file=f)
            for name, func in baselines:
                time.sleep(1)
                latencies = func.profile(N, M, K, A, B, C, warmup=warmup, number=number, repeat=repeat)
                print_latencies(name, latencies)
                print_latencies(name, latencies, file=f)

            for name, allowed in hidet_variants:
                with impl_context(allowed=allowed) as ctx:
                    ir_module = implement(matmul(N, M, K))
                    if use_brute_force_resolve:
                        ir_module = brute_force_resolve(ir_module, warmup=warmup, number=number, repeat=repeat, progress_bar=progress_bar)
                    else:
                        ir_module = random_resolve(ir_module)
                    module = build(ir_module, output_dir=f'./outs/bench/{name}_{N}x{M}x{K}', keep_ir=keep_ir)
                    time.sleep(1)
                    latencies = module['matmul'].profile(A, B, C, warmup=warmup, number=number, repeat=repeat)
                    print_latencies(name, latencies)
                    print_latencies(name, latencies, file=f)
            print()
            print(file=f)


def verify(use_rand=True, keep_ir=False):
    np.set_printoptions(threshold=128 * 128, linewidth=500)
    use_print = True
    workloads = [
        # (1, 1, 1),
        (1, 1, 1),
        # (256, 256, 256),
        # (1234, 2345, 1212),
        # (222, 333, 444),
        (1024, 1024, 1024),
        # (128, 128, 16),
        # (1296, 2304, 768),
        (1296, 2304, 768),
        (1243, 1211, 1207),
        # (1296, 128, 768),
        # (1296, 2304, 768),
        # (1024, 1024, 1024),
        # (2048, 2304, 768),
        *[(16 * T, 2304, 768) for T in [5, 24, 43, 62, 81, 100, 119, 128]]
    ]
    baselines = [
        # ('Opt', matmul_opt()),
    ]
    hidet_variants = [
        # ('HidetNaive', (CudaGridNaiveImplementer, CudaThreadNaiveImplementer)),
        # ('HidetNoPipe', (CudaGridSplitImplementer, CudaBlockStaticMatmulNoPipeImplementer, CudaBlockNaiveImplementer)),
        # ('HidetSoftPipeLdgWb', (CudaGridSplitImplementer, CudaBlockStaticMatmulSoftPipeLdgWbImplementer, CudaBlockNaiveImplementer)),
        # ('HidetSoftPipePred', (CudaGridStaticMatmulSoftPipePredImplementer, CudaBlockNaiveImplementer)),
        ('HidetMatmul', (CudaGridStaticMatmulImplementer,))
    ]
    for N, M, K in workloads:
        print('Workload {} x {} x {}'.format(N, M, K))
        task = matmul(N, M, K)
        if use_rand:
            A = randn([N, K], 'float32', device='cpu')
            B = randn([K, M], 'float32', device='cpu')
        else:
            use_special = True
            if use_special:
                A = np.zeros([N, K], dtype=np.float32)
                B = np.zeros([K, M], dtype=np.float32)
                A[0, 0] = 1.0
                B[0, 0] = 1.0
                A[0, 1] = 1.0
                B[1, 0] = 1.0
                A[1, 1] = 1.0
                B[1, 1] = 1.0
                A = from_numpy(A).cuda()
                B = from_numpy(B).cuda()
            else:
                A = full([N, K], 'float32', 'host', fill_value=1)
                B = full([K, M], 'float32', 'host', fill_value=1)
        C = zeros([N, M], 'float32', 'host')

        for name, baseline in baselines:
            task.worker = Host()
            host_module = build(random_resolve(implement(task)), f'./outs/verify/host/{name}')

            GA, GB, GC = A.cuda(), B.cuda(), C.cuda()
            baseline(N, M, K, GA, GB, GC)

            HA, HB, HC = A.cpu(), B.cpu(), C.cpu()
            host_module['matmul'](HA, HB, HC)
            try:
                np.testing.assert_allclose(GC.numpy(), HC.numpy())
            except AssertionError as e:
                if use_print:
                    print('A:\n{}\nB:\n{}\n{}\n{}\nhost:\n{}'.format(A, B, name, GC, HC))
                raise e

        for name, allowed in hidet_variants:
            print('Verifying {}'.format(name))
            task.worker = Grid()
            with impl_context(allowed=allowed):
                ir_module = implement(task)
                # print(ir_module)
                grid_module = build(random_resolve(ir_module, seed=1), f'./outs/verify/{name}_{N}x{M}x{K}', keep_ir=keep_ir)

            task.worker = Host()
            host_module = build(random_resolve(implement(task)), f'./outs/verify/host/{name}')

            GA, GB, GC = A.cuda(), B.cuda(), C.cuda()
            grid_module['matmul'](GA, GB, GC)
            cuda.device_synchronize()

            HA, HB, HC = A.cpu(), B.cpu(), C.cpu()
            host_module['matmul'](HA, HB, HC)
            try:
                np.testing.assert_allclose(GC.numpy(), HC.numpy())
            except AssertionError as e:
                # use_print = False
                if use_print:
                    print('A:\n{}\nB:\n{}\n{}\n{}\nhost:\n{}'.format(A, B, name, GC, HC))
                raise e


if __name__ == '__main__':
    # verify(keep_ir=False)
    with cuda.BenchmarkContext(lock_clock=False):
        benchmark(use_nsight_compute=False, keep_ir=True)
