import numpy as np

from hidet.backend import build
from hidet.baselines.matmul import matmul_ref, matmul_cublas, matmul_opt, matmul_cutlass
from hidet.implement import implement, impl_context
from hidet.implement.cuda import CudaBlockStaticMatmulSoftPipeLdgWbImplementer
from hidet.implement.cuda import CudaGridSplitImplementer, CudaGridNaiveImplementer, CudaWarpTransfer2dImplementer, CudaWarpFillValueImplementer, CudaBlockStaticMatmulNoPipeImplementer
from hidet.implement.cuda import CudaThreadNaiveImplementer, CudaBlockNaiveImplementer
from hidet.implement.resolve import random_resolve, brute_force_resolve
from hidet.ir.task import Grid, Host
from hidet.runtime.value import TensorValue, randn, empty, scalar, zeros, full
from hidet.tasks.nn import matmul


def print_latencies(name, latencies):
    print('{:>20}: {:.3f} (std {:.3f}) ms [{}]'.format(name, np.mean(latencies), np.std(latencies), " ".join([f'{v:.3f}' for v in latencies])))


def benchmark(warmup=5, number=1, repeat=10, use_brute_force_resolve=False, progress_bar=True, use_nsight_compute=False):
    if use_nsight_compute:
        warmup = 0
        number = 1
        repeat = 1
    workloads = [
        # (2, 2, 2),
        (1024, 1024, 1024),
        # (2048, 2304, 768),
        # (1664, 768, 2304),
    ]
    baselines = [
        ('Reference', matmul_ref()),
        ('Opt', matmul_opt()),
        ('cutlas', matmul_cutlass()),
        ('cuBLAS', matmul_cublas()),
    ]
    hidet_variants = [
        ('HidetNaive', (CudaGridNaiveImplementer, CudaThreadNaiveImplementer)),
        ('HidetNoPipe', (CudaGridSplitImplementer, CudaBlockStaticMatmulNoPipeImplementer, CudaWarpTransfer2dImplementer, CudaBlockNaiveImplementer, CudaWarpFillValueImplementer)),
        ('HidetSoftPipeLdgWb', (CudaGridSplitImplementer, CudaBlockStaticMatmulSoftPipeLdgWbImplementer, CudaWarpTransfer2dImplementer, CudaBlockNaiveImplementer, CudaWarpFillValueImplementer)),
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


def verify(use_rand=True):
    np.set_printoptions(threshold=128 * 128, linewidth=500)
    use_print = True
    workloads = [
        (256, 256, 256),
        # (1024, 1024, 1024),
    ]
    baselines = [
        ('Opt', matmul_opt()),
    ]
    hidet_variants = [
        ('HidetNaive', (CudaGridNaiveImplementer, CudaThreadNaiveImplementer)),
        ('HidetNoPipe', (CudaGridSplitImplementer, CudaBlockStaticMatmulNoPipeImplementer, CudaWarpTransfer2dImplementer, CudaBlockNaiveImplementer, CudaWarpFillValueImplementer)),
        ('HidetSoftPipeLdgWb', (CudaGridSplitImplementer, CudaBlockStaticMatmulSoftPipeLdgWbImplementer, CudaWarpTransfer2dImplementer, CudaBlockNaiveImplementer, CudaWarpFillValueImplementer)),
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
                A[0, 1] = 1.0
                B[1, 0] = 1.0
                A[1, 1] = 1.0
                B[1, 1] = 1.0
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
            try:
                np.testing.assert_allclose(GC.to_numpy(), HC.to_numpy())
            except AssertionError as e:
                if use_print:
                    print('A:\n{}\nB:\n{}\n{}\n{}\nhost:\n{}'.format(A, B, name, GC, HC))
                raise e

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
            try:
                np.testing.assert_allclose(GC.to_numpy(), HC.to_numpy())
            except AssertionError as e:
                if use_print:
                    print('A:\n{}\nB:\n{}\n{}\n{}\nhost:\n{}'.format(A, B, name, GC, HC))
                raise e


def test_demo():
    N, M, K = 2, 2, 2
    task = matmul(N, M, K)
    ir_module = implement(task)
    ir_module = random_resolve(ir_module)
    module = build(ir_module, output_dir='./outs')

    A = TensorValue.randn([N, K], 'float32', 'global', seed=1)
    B = TensorValue.randn([K, M], 'float32', 'global', seed=3)
    C = TensorValue.empty([N, M], 'float32', 'global')
    module['matmul'](A, B, C)


if __name__ == '__main__':
    # verify()
    benchmark(use_nsight_compute=False)
    # test_demo()

    # from hidet.ir import *
    # from hidet.ir.layout import *
    # # ((((((threadIdx.x / 32) / 2) * 16) + (((((threadIdx.x % 32) / 16) * 2) + ((threadIdx.x % 32) % 2)) * 4)) + 3) % 16)
    # with FunctionBuilder('test', attrs={'worker': ThreadBlock(block_dim=256)}) as fb:
    #     arr = Var('arr', TensorType(scope='register', dtype='float32', layout=DataLayout.row_major([100])))
    #     fb.extend_local_vars([arr])
    #     sb = StmtBuilder()
    #     with sb.let('v', thread_idx()) as v:
    #         # sb += BufferStoreStmt(arr, [((((((v // 32) // 2) * 16) + (((((v % 32) // 16) * 2) + ((v % 32) % 2)) * 4)) + 3) % 16)], convert(0.0))
    #         # sb += BufferStoreStmt(arr, [(((v * 32) + (v % 32)) // 32)], convert(0.0))
    #         # sb += BufferStoreStmt(arr, [(((v // 64) * 16) // 16)], convert(0.0))
    #         sb += BufferStoreStmt(arr, [((((((((((v / (32 * 2)) % 4) * 32) + ((0 * 4) + (((((((v % 32) / 1) % 32) / 16) * 2) + ((((v % 32) / 1) % 32) % 2)) * 4))) + 0) / (4 * 4)) % 2) * 2) + ((((((((v / (32 * 1)) % 2) * 64) + ((8 * 4) + ((((((v % 32) / 1) % 32) / 2) % 8) * 4))) + 2) / (4 * 8)) % 2) * 1)) + 0)], convert(0.0))
    #
    #     fb.set_body(sb.finish())
    # ir_module = IRModule()
    # ir_module.add('test', fb.get())
    # module = build(ir_module, output_dir='./outs/test')
    # with open('./outs/test/source.cu', 'r') as f:
    #     print("\n".join(f.readlines()))
