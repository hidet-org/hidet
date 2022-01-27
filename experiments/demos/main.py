import numpy as np

from hidet.backend import build
from hidet.baselines.matmul import matmul_ref, matmul_cublas, matmul_opt, matmul_cutlass
from hidet.implement import implement, impl_context
from hidet.implement.cuda import CudaBlockStaticMatmulSoftPipeImplementer, CudaBlockStaticMatmulNoPipeImplementer, CudaBlockStaticMatmulNoPipeLdgImplementer, CudaBlockStaticMatmulSoftPipeLdgImplementer
from hidet.implement.cuda import CudaGridSplitImplementer, CudaGridNaiveImplementer, CudaWarpTransfer2dImplementer, CudaBlockTransfer2dImplementer, CudaWarpMmaImplementer, CudaWarpFillValueImplementer
from hidet.implement.cuda import CudaThreadNaiveImplementer
from hidet.implement.resolve import random_resolve, brute_force_resolve
from hidet.ir.task import Grid
from hidet.ir.task import Host
from hidet.nn import matmul
from hidet.runtime.value import TensorValue, randn, empty, scalar, zeros, full


def print_latencies(name, latencies):
    print('{:>20}: {:.3f} (std {:.3f}) ms [{}]'.format(name, np.mean(latencies), np.std(latencies), " ".join([f'{v:.3f}' for v in latencies])))


def benchmark(warmup=5, number=1, repeat=10, use_brute_force_resolve=True, progress_bar=True):
    # warmup = 0
    # number = 1
    # repeat = 1
    # use_brute_force_resolve = True
    workloads = [
        (1024, 1024, 1024),
        (1664, 768, 2304),
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
    print('Repeat = {}'.format(repeat))
    print('Brute-force resolver = {}'.format(use_brute_force_resolve))
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


def verify(use_rand=False):
    np.set_printoptions(threshold=128 * 128, linewidth=500)
    workloads = [
        (256, 256, 256),
        # (1600, 768, 2304)
        # (128, 128, 8),
        # (4 * 2, 8 * 2, 8 * 2),
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
    for N, M, K in workloads:
        print('Workload {} x {} x {}'.format(N, M, K))
        task = matmul(N, M, K)
        if use_rand:
            A = randn([N, K], 'float32', 'host', seed=1)
            B = randn([K, M], 'float32', 'host', seed=3)
        else:
            use_special = True
            if use_special:
                A = np.zeros([N, K], dtype=np.float32)
                B = np.zeros([K, M], dtype=np.float32)
                A[0, 0] = 1.0
                B[0, 0] = 1.0
                A = TensorValue.from_numpy(A, scope='global')
                B = TensorValue.from_numpy(B, scope='global')
            else:
                A = full([N, K], 'float32', 'host', fill_value=1)
                B = full([K, M], 'float32', 'host', fill_value=1)
        C = zeros([N, M], 'float32', 'host')

        for name, baseline in baselines:
            print('Verifying {}'.format(name))
            task.worker = Host()
            host_module = build(random_resolve(implement(task)), f'./outs/verify/host/{name}')

            GA, GB, GC = A.to_cuda(), B.to_cuda(), C.to_cuda()
            baseline(scalar(N), scalar(M), scalar(K), GA, GB, GC)

            HA, HB, HC = A.to_cpu(), B.to_cpu(), C.to_cpu()
            host_module['matmul'](HA, HB, HC)
            np.testing.assert_allclose(GC.to_numpy(), HC.to_numpy())

        for name, allowed in hidet_variants:
            print('Verifying {}'.format(name))
            task.worker = Grid()
            with impl_context(allowed=allowed):
                ir_module = implement(task)
                grid_module = build(random_resolve(ir_module, seed=1), f'./outs/verify/{name}')

            task.worker = Host()
            host_module = build(random_resolve(implement(task)), f'./outs/verify/host/{name}')

            GA, GB, GC = A.to_cuda(), B.to_cuda(), C.to_cuda()
            grid_module['matmul'](GA, GB, GC)

            HA, HB, HC = A.to_cpu(), B.to_cpu(), C.to_cpu()
            host_module['matmul'](HA, HB, HC)
            np.testing.assert_allclose(GC.to_numpy(), HC.to_numpy())


if __name__ == '__main__':
    # benchmark()
    verify()
