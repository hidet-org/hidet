import numpy as np
import time
import os

from hidet.backend import build
from hidet.baselines.matmul import matmul_cublas, matmul_opt, matmul_cutlass, matmul_cublas_tensorcore
from hidet.implement import implement, impl_context
from hidet.implement.cuda import CudaGridStaticMatmulImplementer
from hidet.implement.resolve import random_resolve, brute_force_resolve
from hidet.ir.task import Grid, Host, ThreadBlock
from hidet.ir.type import TensorType
from hidet.ir.expr import Var, convert
from hidet.ir.stmt import BufferStoreStmt
from hidet.ir.func import IRModule
from hidet.ir.primitives import thread_idx, block_idx
from hidet.ir.layout import DataLayout
from hidet.runtime.value import TensorValue, randn, empty, scalar, zeros, full
from hidet.tasks.nn import matmul
from hidet.ir.builders import FunctionBuilder, StmtBuilder
from hidet.utils import cuda


def print_latencies(name, latencies, file=None):
    print('{:>20}: {:.3f} (std {:.3f}) ms [{}]'.format(name, np.median(latencies), np.std(latencies), " ".join([f'{v:.3f}' for v in latencies])), file=file)


def benchmark(warmup=5, number=1, repeat=10, use_brute_force_resolve=False, progress_bar=True, use_nsight_compute=False, keep_ir=False):
    if use_nsight_compute:
        warmup = 0
        number = 1
        repeat = 1
    workloads = [
        # (2, 2, 2),
        (222, 333, 444),
        (1024, 1024, 1024),
        (1024 + 22, 1024 + 33, 1024 + 44),
        # (1296, 2304, 768),
        # (1243, 1211, 1207),
        # (1024 + 128, 1024 + 128, 1024 + 48),
        # (2048, 2304, 768),
        # (1664, 768, 2304),
        # (1234, 2345, 1212),
        *[(16 * T, 2304, 768) for T in [5, 24, 43, 62, 81, 100, 119, 128]],
        # *[(16 * T, 2304, 768) for T in [5, 81, 128]]
    ]
    baselines = [
        # ('Reference', matmul_ref()),
        ('Opt', matmul_opt()),
        ('cutlas', matmul_cutlass()),
        ('cublas', matmul_cublas()),
        # ('cublas_tc', matmul_cublas_tensorcore()),
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
            A = randn([N, K], 'float32', 'global', seed=1)
            B = randn([K, M], 'float32', 'global', seed=3)
            C = empty([N, M], 'float32', 'global')
            print("Workload (N x M x K): {} x {} x {}".format(N, M, K))
            print("Workload (N x M x K): {} x {} x {}".format(N, M, K), file=f)
            for name, func in baselines:
                time.sleep(1)
                latencies = func.profile(scalar(N), scalar(M), scalar(K), A, B, C, warmup=warmup, number=number, repeat=repeat)
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
                # print(ir_module)
                grid_module = build(random_resolve(ir_module, seed=1), f'./outs/verify/{name}_{N}x{M}x{K}', keep_ir=keep_ir)

            task.worker = Host()
            host_module = build(random_resolve(implement(task)), f'./outs/verify/host/{name}')

            GA, GB, GC = A.to_cuda(), B.to_cuda(), C.to_cuda()
            grid_module['matmul'](GA, GB, GC)
            cuda.device_synchronize()

            HA, HB, HC = A.to_cpu(), B.to_cpu(), C.to_cpu()
            host_module['matmul'](HA, HB, HC)
            try:
                np.testing.assert_allclose(GC.to_numpy(), HC.to_numpy())
            except AssertionError as e:
                # use_print = False
                if use_print:
                    print('A:\n{}\nB:\n{}\n{}\n{}\nhost:\n{}'.format(A, B, name, GC, HC))
                raise e


def test_custom_func():
    # ((((((threadIdx.x / 32) / 2) * 16) + (((((threadIdx.x % 32) / 16) * 2) + ((threadIdx.x % 32) % 2)) * 4)) + 3) % 16)
    with FunctionBuilder('test', attrs={'worker': ThreadBlock(block_dim=256)}) as fb:
        arr = Var('arr', TensorType(scope='register', dtype='float32', layout=DataLayout.row_major([100])))
        fb.extend_local_vars([arr])
        sb = StmtBuilder()
        # with sb.let('v', thread_idx()) as v:
        with sb.for_loop('v', extent=256) as threadIdx:
            with sb.for_loop('vv', extent=1000) as blockIdx:
                threadIdx = thread_idx()
                blockIdx = block_idx()
                with sb.for_loop('j', extent=2) as j:
                    # with sb.let('a', 1) as a:
                    #     with sb.let('b', 2) as b:
                    #         sb += BufferStoreStmt(arr, [a + b], convert(0.0))
                    # with sb.let('c', 1) as c:
                    #     sb += BufferStoreStmt(arr, [c], convert(0.0))
                    # sb += BufferStoreStmt(arr, [((((((v // 32) // 2) * 16) + (((((v % 32) // 16) * 2) + ((v % 32) % 2)) * 4)) + 3) % 16)], convert(0.0))
                    # sb += BufferStoreStmt(arr, [(((v * 32) + (v % 32)) // 32)], convert(0.0))
                    # sb += BufferStoreStmt(arr, [(((v // 64) * 16) // 16)], convert(0.0))
                    # sb += BufferStoreStmt(arr, [v + (convert(0) + convert(0))], convert(0.0))
                    # sb += BufferStoreStmt(arr, [((0 + ((0 + ((1 * (v / 8)) * 128)) * 1024)) + (0 * 1))], convert(0.0))
                    # sb += BufferStoreStmt(arr, [(((((((v % 32) / 16) * 2) + ((v % 32) % 2)) * 4) + ((v / 64) * 16)) % 16)], convert(0.0))
                    # sb += BufferStoreStmt(arr, [((thread_idx() % 32) % 2)], convert(0.0))
                    with sb.if_then(((blockIdx / 18) * 32) + ((threadIdx / 64) * 16) < 1296):
                        sb += BufferStoreStmt(arr, [(((((blockIdx / 18) * 32) + ((threadIdx / 64) * 16)) * 2304) + (((blockIdx % 18) * 128) + ((((threadIdx / 32) % 2) * 64) + ((j * 32) + (threadIdx % 32)))))], arr[(((((threadIdx / 64) * 2) + ((threadIdx / 32) % 2)) * 512) + (threadIdx % 32))])
                    with sb.if_then(((blockIdx / 18) * 32) + (((threadIdx / 64) * 16) + 1) < 1296):
                        sb += BufferStoreStmt(arr, [((((((threadIdx / 64) * 16) + ((blockIdx / 18) * 32)) * 2304) + (((blockIdx % 18) * 128) + ((((threadIdx / 32) % 2) * 64) + ((j * 32) + (threadIdx % 32))))) + 2304)], arr[(((threadIdx % 32) + ((((threadIdx / 64) * 2) + ((threadIdx / 32) % 2)) * 512)) + 32)])
        fb.set_body(sb.finish())
    ir_module = IRModule()
    ir_module.add('test', fb.get())
    module = build(ir_module, output_dir='./outs/test', keep_ir=True)
    with open('./outs/test/source.cu', 'r') as f:
        print("".join(f.readlines()))


if __name__ == '__main__':
    # verify(keep_ir=False)
    with cuda.BenchmarkContext(fix_clock=True):
        benchmark(use_nsight_compute=False, keep_ir=False)
    # test_custom_func()
