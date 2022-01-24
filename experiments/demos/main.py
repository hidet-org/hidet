import os

import numpy as np
import pycuda.driver
import sympy

from hidet.backend import codegen, build
from hidet.baselines.matmul import matmul_ref, matmul_cublas, matmul_opt, matmul_cutlass
from hidet.implement import implement, impl_context
from hidet.implement.resolve import random_resolve, brute_force_resolve
from hidet.ir.dialects.compute import tensor_input, reduce_sum, compute
from hidet.ir.expr import var
from hidet.ir.functors import astext
from hidet.ir.task import Task, Grid, Host
from hidet.ir.type import tensor_type
from hidet.nn import matmul
from hidet.runtime.value import TensorValue, randn, empty, scalar, zeros, full
from hidet.testing import verify
from hidet.transforms import flatten_tensor_pass, generate_packed_func_pass
from hidet.implement.cuda import CudaGridSplitImplementer, CudaGridNaiveImplementer
from hidet.implement.cuda import CudaBlockStaticMatmulSoftPipeImplementer, CudaBlockStaticMatmulNoPipeImplementer, CudaBlockNaiveImplementer


def get_task(N=1024, M=1024, K=1024):
    k = var('k')

    A = tensor_input('A', 'float32', [N, K])
    B = tensor_input('B', 'float32', [K, M])
    C = compute('C', [N, M], lambda i, j: reduce_sum(A[i, k] * B[k, j], axis=k, shape=[K]))

    params_type = [
        tensor_type('global', 'float32', [N, K], [K, 1]),
        tensor_type('global', 'float32', [K, M], [M, 1]),
        tensor_type('global', 'float32', [N, M], [M, 1])
    ]
    task = Task('gemm', C, [A, B, C], params_type, Grid())
    return task


def demo_task():
    task = get_task()
    module = implement(task)
    print(astext(module))


def demo_codegen():
    task = get_task()
    module = implement(task)
    print(codegen(module))


def demo_split():
    task = get_task()
    ir_module = implement(task)
    ir_module = generate_packed_func_pass()(ir_module)
    ir_module = flatten_tensor_pass()(ir_module)
    # print(astext(ir_module))
    print(codegen(ir_module))


def demo_build():
    task = get_task()
    ir_module = implement(task)
    ir_module = generate_packed_func_pass()(ir_module)
    ir_module = flatten_tensor_pass()(ir_module)
    target_dir = './test_task'
    os.makedirs(target_dir, exist_ok=True)
    module = build(ir_module, target_dir)
    A = TensorValue.empty([1024, 1024], 'float32', 'global')
    B = TensorValue.empty([1024, 1024], 'float32', 'global')
    C = TensorValue.empty([1024, 1024], 'float32', 'global')
    module['gemm'](A, B, C)


def demo_test():
    N = 2
    M = 2
    K = 2
    task = matmul(N, M, K)
    ir_module = implement(task)
    module = build(ir_module, output_dir='./outs')
    A = randn([N, K], 'float32', 'global', seed=1)
    B = randn([K, M], 'float32', 'global', seed=3)
    C = empty([N, M], 'float32', 'global')
    module['matmul'](A, B, C)
    print(A)
    print(B)
    print(C)


def demo_matmul():
    N, M, K = 512, 512, 512
    ir_module = implement(matmul(N, M, K))
    module = build(ir_module, output_dir='./outs')

    A = randn([N, K], 'float32', 'global', seed=1)
    B = randn([K, M], 'float32', 'global', seed=3)
    C = empty([N, M], 'float32', 'global')
    module['matmul'](A, B, C)
    print(A)
    print(B)
    print(C)


def demo_profile():
    N, M, K = 1024, 1024, 1024
    ir_module = implement(matmul(N, M, K), 'cuda_grid_split_implementer')
    module = build(ir_module, output_dir='./outs')

    A = randn([N, K], 'float32', 'global', seed=1)
    B = randn([K, M], 'float32', 'global', seed=3)
    C = empty([N, M], 'float32', 'global')
    print(module['matmul'].profile(A, B, C, repeat=10))


def print_latencies(name, latencies):
    print('{:>20}: {:.3f} (std {:.3f}) ms [{}]'.format(name, np.mean(latencies), np.std(latencies), " ".join([f'{v:.3f}' for v in latencies])))


def demo_baselines():
    warmup = 5
    number = 1
    repeat = 20
    use_brute_force_resolve = True
    workloads = [
        (1024, 1024, 1024),
        # (1600, 768, 2304)
        # (128, 128, 16),
    ]
    baselines = [
        ('Reference', matmul_ref()),
        ('Opt', matmul_opt()),
        ('cutlas', matmul_cutlass()),
        ('cuBLAS', matmul_cublas()),
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

        with impl_context(try_first=[CudaGridSplitImplementer, CudaBlockStaticMatmulNoPipeImplementer]):
            ir_module = implement(matmul(N, M, K))
            if use_brute_force_resolve:
                ir_module = brute_force_resolve(ir_module, warmup=warmup, number=number, repeat=repeat, progress_bar=False)
            else:
                ir_module = random_resolve(ir_module)
            module = build(ir_module, output_dir='./outs/no_pipe')
            latencies = module['matmul'].profile(A, B, C, warmup=warmup, number=number, repeat=repeat)
            print_latencies('hidet_no_pipe', latencies)

        with impl_context(try_first=[CudaGridSplitImplementer, CudaBlockStaticMatmulSoftPipeImplementer]):
            ir_module = implement(matmul(N, M, K))
            if use_brute_force_resolve:
                ir_module = brute_force_resolve(ir_module, warmup=warmup, number=number, repeat=repeat, progress_bar=False)
            else:
                ir_module = random_resolve(ir_module)
            module = build(ir_module, output_dir='./outs/soft_pipe')
            latencies = module['matmul'].profile(A, B, C, warmup=warmup, number=number, repeat=repeat)
            print_latencies('hidet_soft_pipe', latencies)

        with impl_context(try_first=[CudaGridNaiveImplementer, CudaBlockNaiveImplementer]):
            ir_module = implement(matmul(N, M, K))
            if use_brute_force_resolve:
                ir_module = brute_force_resolve(ir_module, warmup=warmup, number=number, repeat=repeat, progress_bar=False)
            else:
                ir_module = random_resolve(ir_module)
            module = build(ir_module, output_dir='./outs/naive')
            latencies = module['matmul'].profile(A, B, C, warmup=warmup, number=number, repeat=repeat)
            print_latencies('hidet_naive', latencies)

        with impl_context(try_first=[CudaGridSplitImplementer, CudaBlockNaiveImplementer]):
            ir_module = implement(matmul(N, M, K))
            if use_brute_force_resolve:
                ir_module = brute_force_resolve(ir_module, warmup=warmup, number=number, repeat=repeat, progress_bar=False)
            else:
                ir_module = random_resolve(ir_module)
            module = build(ir_module, output_dir='./outs/naive_split')
            latencies = module['matmul'].profile(A, B, C, warmup=warmup, number=number, repeat=repeat)
            print_latencies('hidet_naive_split', latencies)

        print()


def demo_host():
    N, M, K = 2, 2, 2
    k = var('k')

    A = tensor_input('A', 'float32', [N, K])
    B = tensor_input('B', 'float32', [K, M])
    C = compute('C', [N, M], lambda i, j: reduce_sum(A[i, k] * B[k, j], axis=k, shape=[K]))

    params_type = [
        tensor_type('global', 'float32', [N, K], [K, 1]),
        tensor_type('global', 'float32', [K, M], [M, 1]),
        tensor_type('global', 'float32', [N, M], [M, 1])
    ]
    task = Task('gemm', C, [A, B, C], params_type, Host())
    ir_module = implement(task)
    module = build(ir_module, output_dir='./outs')
    A = randn([N, K], 'float32', 'host', seed=1)
    B = randn([K, M], 'float32', 'host', seed=3)
    C = empty([N, M], 'float32', 'host')
    module['gemm'](A, B, C)
    print(A)
    print(B)
    print(C)


def demo_verify():
    # for workload in [(4 * 2, 8 * 3, 8 * 4)]:
    for workload in [(1024, 1024, 1024)]:
        # N, M, K = V + 22, V*2 - 3, V // 2 + 3
        N, M, K = workload
        task = matmul(N, M, K)
        # A = randn([N, K], 'float32', 'host', seed=1)
        # B = randn([K, M], 'float32', 'host', seed=3)
        A = full([N, K], 'float32', 'host', fill_value=1)
        B = full([K, M], 'float32', 'host', fill_value=1)
        C = zeros([N, M], 'float32', 'host')

        # A = np.full(shape=[N, K], fill_value=0.0, dtype=np.float32, order='C')
        # B = np.full(shape=[K, M], fill_value=0.0, dtype=np.float32, order='C')
        # C = np.full(shape=[N, M], fill_value=0.0, dtype=np.float32, order='C')
        # A[:, 0] = 1.0
        # B[0, :] = 1.0
        # A = TensorValue.from_numpy(A, scope='host')
        # B = TensorValue.from_numpy(B, scope='host')
        # C = TensorValue.from_numpy(C, scope='host')

        np.set_printoptions(threshold=128 * 128, linewidth=500)
        use_verify = False
        if use_verify:
            verify(task, [A, B, C], grid_implementor='cuda_grid_split_implementer')
        else:
            task.worker = Grid()

            with impl_context(try_first='cuda_grid_split_implementer'):
                ir_module = implement(task)
                grid_module = build(random_resolve(ir_module, seed=1), f'./outs/pipe')
                # grid_module = build(random_resolve(implement(task, impl_name='cuda_grid_naive_implementer'), seed=1), f'./outs/grid')

            task.worker = Host()
            host_module = build(random_resolve(implement(task)), f'./outs/host')

            GA, GB, GC = A.to_cuda(), B.to_cuda(), C.to_cuda()
            grid_module['matmul'](GA, GB, GC)
            pycuda.driver.Context.synchronize()
            # print(GA)
            # print(GB)
            # print(GC)

            HA, HB, HC = A.to_cpu(), B.to_cpu(), C.to_cpu()
            host_module['matmul'](HA, HB, HC)
            # print(HC)
            np.testing.assert_allclose(GC.to_numpy(), HC.to_numpy())


def demo_grid_2d_static_implementer():
    N, M, K = 2, 2, 2
    with impl_context(try_first='cuda_grid_split_implementer'):
        ir_module = implement(matmul(N, M, K))
    ir_module = random_resolve(ir_module)
    module = build(ir_module, output_dir='./outs')

    A = randn([N, K], 'float32', 'global', seed=1)
    B = randn([K, M], 'float32', 'global', seed=3)
    C = empty([N, M], 'float32', 'global')
    module['matmul'](A, B, C)
    print(A)
    print(B)
    print(C)


def demo_sympy():
    a, b = sympy.symbols('a b')
    expr = (a + b) * (a + b) - a * a - 2 * a * b - b * b
    print(expr)
    print(sympy.simplify(expr), type(sympy.simplify(expr)))
    print(sympy.simplify(expr) == 0)


def demo_brute_force_resolver():
    demo_baselines()
    workloads = [
        (1024, 1024, 1024),
        # (1600, 768, 2304)
    ]

    for workload in workloads:
        N, M, K = workload
        with impl_context(try_first='cuda_grid_split_implementer'):
            ir_module = implement(matmul(N, M, K))
        ir_module = brute_force_resolve(ir_module, repeat=10)
        # ir_module = random_resolve(ir_module, seed=1)
        module = build(ir_module, output_dir='./outs')

        A = randn([N, K], 'float32', 'global', seed=1)
        B = randn([K, M], 'float32', 'global', seed=3)
        C = empty([N, M], 'float32', 'global')
        print(np.mean(module['matmul'].profile(A, B, C, repeat=10)))


def demo_efficient_matmul():
    N, M, K = 1024, 1024, 1024
    with impl_context(try_first='cuda_grid_split_implementer'):
        ir_module = random_resolve(implement(matmul(N, M, K)))
    module = build(ir_module, './outs/no_pipe')


if __name__ == '__main__':
    # demo_task()
    # demo_codegen()
    # demo_split()
    # demo_build()
    # demo_test()
    # demo_profile()
    demo_baselines()
    # demo_host()
    # demo_verify()
    # demo_grid_2d_static_implementer()
    # demo_sympy()
    # demo_brute_force_resolver()
    # demo_efficient_matmul()
    # demo_verify()
