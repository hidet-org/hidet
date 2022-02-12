import numpy as np
import ctypes
import os

from hidet.backend import build
from hidet.baselines.matmul import matmul_ref, matmul_cublas, matmul_opt, matmul_cutlass
from hidet.implement import implement, impl_context
from hidet.implement.cuda import CudaBlockStaticMatmulNoPipeImplementer, CudaBlockStaticMatmulSoftPipeLdgWbImplementer
from hidet.implement.cuda import CudaGridSplitImplementer, CudaGridNaiveImplementer, CudaWarpTransfer2dImplementer, CudaWarpMmaImplementer, CudaWarpFillValueImplementer
from hidet.implement.cuda import CudaThreadNaiveImplementer, CudaBlockNaiveImplementer
from hidet.implement.resolve import random_resolve, brute_force_resolve
from hidet.ir.func import IRModule
from hidet.ir.task import Grid, Host
from hidet.runtime.value import TensorValue, randn, empty, scalar, zeros, full
from hidet.runtime.module import CompiledModule, CompiledFunction
from hidet.ir.dialects.lowlevel import VoidType
from hidet.ffi import PackedFunc
from hidet.tasks.nn import matmul
from hidet.backend.build import lower, codegen, compile_src_code


def print_latencies(name, latencies):
    print('{:>20}: {:.3f} (std {:.3f}) ms [{}]'.format(name, np.mean(latencies), np.std(latencies), " ".join([f'{v:.3f}' for v in latencies])))


def benchmark(warmup=5, number=1, repeat=10, use_brute_force_resolve=False, progress_bar=True):
    use_nsight_compute = True
    if use_nsight_compute:
        warmup = 0
        number = 1
        repeat = 1
    workloads = [
        (1024, 1024, 1024),
        (2048, 2304, 768),
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
        ('HidetNoPipe', (CudaGridSplitImplementer, CudaBlockStaticMatmulNoPipeImplementer, CudaWarpTransfer2dImplementer, CudaBlockNaiveImplementer, CudaWarpMmaImplementer, CudaWarpFillValueImplementer)),
        ('HidetSoftPipeLdgWb', (CudaGridSplitImplementer, CudaBlockStaticMatmulSoftPipeLdgWbImplementer, CudaWarpTransfer2dImplementer, CudaBlockNaiveImplementer, CudaWarpMmaImplementer, CudaWarpFillValueImplementer)),
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
        # (16, 16, 2),
        # (16, 16, 4),
        # (128, 128, 16),
        (256, 256, 256),
        # (1600, 768, 2304)
        # (128, 128, 8),
        # (4 * 2, 8 * 2, 8 * 2),
    ]
    baselines = [
        ('Opt', matmul_opt()),
    ]
    hidet_variants = [
        ('HidetNaive', (CudaGridNaiveImplementer, CudaThreadNaiveImplementer)),
        # ('HidetNoPipe', (CudaGridSplitImplementer, CudaBlockStaticMatmulNoPipeImplementer, CudaWarpTransfer2dImplementer, CudaBlockNaiveImplementer, CudaWarpMmaImplementer, CudaWarpFillValueImplementer)),
        # ('HidetNoPipeLdg', (CudaGridSplitImplementer, CudaBlockStaticMatmulNoPipeLdgImplementer, CudaWarpTransfer2dImplementer, CudaBlockNaiveImplementer, CudaWarpMmaImplementer, CudaWarpFillValueImplementer)),
        # ('HidetSoftPipe', (CudaGridSplitImplementer, CudaBlockStaticMatmulSoftPipeImplementer, CudaWarpTransfer2dImplementer, CudaBlockNaiveImplementer, CudaWarpMmaImplementer, CudaWarpFillValueImplementer)),
        # ('HidetSoftPipeLdg', (CudaGridSplitImplementer, CudaBlockStaticMatmulSoftPipeLdgImplementer, CudaWarpTransfer2dImplementer, CudaBlockNaiveImplementer, CudaWarpMmaImplementer, CudaWarpFillValueImplementer)),
        ('HidetSoftPipeLdgWb', (CudaGridSplitImplementer, CudaBlockStaticMatmulSoftPipeLdgWbImplementer, CudaWarpTransfer2dImplementer, CudaBlockNaiveImplementer, CudaWarpMmaImplementer, CudaWarpFillValueImplementer)),
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


def build_given_src(ir_module: IRModule, src_path, output_dir, keep=True) -> CompiledModule:
    # lower
    ir_module = lower(ir_module)

    # codegen
    os.makedirs(output_dir, exist_ok=True)
    src_code_not_used, func_name_map = codegen(ir_module)

    # call target compiler to get dynamic library
    lib_path = compile_src_code(src_path, keep=keep)

    # load dynamic library
    lib = ctypes.CDLL(lib_path)
    compiled_funcs = {}
    for func in ir_module.functions.values():
        # only load the packed function into python CompiledFunction
        if func.get_attr('packed_func') is not None:
            assert isinstance(func.ret_type, VoidType)
            target_func = ir_module.lookup(func.get_attr('packed_func'))
            target_func_param_types = [p.type for p in target_func.params]
            packed_func = PackedFunc(target_func_param_types, lib[func_name_map[func.name]])
            compiled_funcs[func.name] = CompiledFunction(func.name, func, packed_func)

    return CompiledModule(ir_module, compiled_funcs, None)


if __name__ == '__main__':
    # verify()
    benchmark()
