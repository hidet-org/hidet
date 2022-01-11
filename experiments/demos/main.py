import numpy as np
import sympy
import os
from hidet.ir.type import tensor_type
from hidet.ir.task import Task, Grid, Host
from hidet.ir.functors import astext
from hidet.ir.dialects.compute import Axis, tensor_input, reduce_sum, compute
from hidet.transforms import const_expr_simplifier_pass, flatten_tensor_pass, generate_packed_func_pass
from hidet.runtime.value import TensorValue, randn, empty, scalar, zeros
from hidet.implement import implement
from hidet.implement.resolve import random_resolve
from hidet.nn import matmul
from hidet.baselines.matmul import matmul_ref, matmul_cublas, matmul_opt, matmul_cutlass
from hidet.backend import codegen, build, lower
from hidet.testing import verify


def get_task(N=1024, M=1024, K=1024):
    k = Axis(K)

    A = tensor_input('A', 'float32', [N, K])
    B = tensor_input('B', 'float32', [K, M])
    C = compute('C', [N, M], lambda i, j: reduce_sum(A[i, k] * B[k, j], axis=k))

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
    task = get_task(N, M, K)
    ir_module = implement(task)
    module = build(ir_module, output_dir='./outs')
    A = randn([N, K], 'float32', 'global', seed=1)
    B = randn([K, M], 'float32', 'global', seed=3)
    C = empty([N, M], 'float32', 'global')
    module['gemm'](A, B, C)
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
    ir_module = implement(matmul(N, M, K))
    module = build(ir_module, output_dir='./outs')

    A = randn([N, K], 'float32', 'global', seed=1)
    B = randn([K, M], 'float32', 'global', seed=3)
    C = empty([N, M], 'float32', 'global')
    print(module['matmul'].profile(A, B, C, repeat=10))


def demo_baselines():
    repeat = 10
    workloads = [
        (1024, 1024, 1024),
        (1600, 768, 2304)
    ]
    baselines = [
        ('Reference', matmul_ref()),
        ('Opt', matmul_opt()),
        ('cutlas', matmul_cutlass()),
        ('cuBLAS', matmul_cublas())
    ]
    print('Repeat = {}'.format(repeat))
    print()
    for N, M, K in workloads:
        A = randn([N, K], 'float32', 'global', seed=1)
        B = randn([K, M], 'float32', 'global', seed=3)
        C = empty([N, M], 'float32', 'global')
        print("Workload (N x M x K): {} x {} x {}".format(N, M, K))
        for name, func in baselines:
            latencies = func.profile(scalar(N), scalar(M), scalar(K), A, B, C, repeat=repeat)
            print('{:>13}: {:.3f} (std {:.3f}) ms'.format(name, np.mean(latencies), np.std(latencies)))

        module = build(implement(matmul(N, M, K)), output_dir='./outs')
        latencies = module['matmul'].profile(A, B, C, repeat=repeat)
        print('{:>13}: {:.3f} (std {:.3f}) ms'.format('hidet', np.mean(latencies), np.std(latencies)))

        print()


def demo_host():
    N, M, K = 2, 2, 2
    k = Axis(K)

    A = tensor_input('A', 'float32', [N, K])
    B = tensor_input('B', 'float32', [K, M])
    C = compute('C', [N, M], lambda i, j: reduce_sum(A[i, k] * B[k, j], axis=k))

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
    for V in [2, 4, 8, 16, 32, 64, 128, 256, 512]:
        # N, M, K = V + 22, V*2 - 3, V // 2 + 3
        N, M, K = V, V, V
        task = matmul(N, M, K)
        A = randn([N, K], 'float32', 'host', seed=1)
        B = randn([K, M], 'float32', 'host', seed=3)
        C = zeros([N, M], 'float32', 'host')

        use_verify = False
        if use_verify:
            verify(task, [A, B, C], grid_implementor='cuda_grid_split_implementer')
        else:
            task.worker = Grid()
            grid_module = build(random_resolve(implement(task, impl_name='cuda_grid_split_implementer'), seed=1), f'./outs/grid')

            task.worker = Host()
            host_module = build(random_resolve(implement(task)), f'./outs/host')

            GA, GB, GC = A.to_cuda(), B.to_cuda(), C.to_cuda()
            grid_module['matmul'](GA, GB, GC)
            print(GA)
            print(GB)
            print(GC)

            HA, HB, HC = A.to_cpu(), B.to_cpu(), C.to_cpu()
            host_module['matmul'](HA, HB, HC)
            print(HC)
            np.testing.assert_allclose(GC.to_numpy(), HC.to_numpy())


def demo_grid_2d_static_implementer():
    N, M, K = 2, 2, 2
    ir_module = implement(matmul(N, M, K), impl_name='cuda_grid_split_implementer')
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


if __name__ == '__main__':
    # demo_task()
    # demo_codegen()
    # demo_split()
    # demo_build()
    # demo_test()
    # demo_profile()
    # demo_baselines()
    # demo_host()
    demo_verify()
    # demo_grid_2d_static_implementer()
    # demo_sympy()

"""
TOS: Task-Oriented Scheduling
Hidet: HIerarchical DEcomposable Task

Task
    worker: grid threadblock warp thread
    memory scope: global shared register
    compute workload: out[i, j, k] = ...


matmul
batch_matmul
conv
depthwise conv
pointwise conv
separable conv

----------------
Target Workloads

for a in grid(A):
    output[A] = reduce(expr(inputs[lc(a, r)]) for r in grid(R))

for a in grid(A):
    output[A] = init_value
    for r in grid(R):
        expr(output[a], input_1[lc_1(a, r)], ...)

for ao in grid(AO):
    output[ao + ai] = init_value
    for ro in grid(RO):
        for ai grid(AI):
            for ri in grid(RI):
                expr(output[ao + ai], input_1[lc_1(ao, ai, ro, ri)], ...)


*** 
matmul(N, M, K):
for i in range(N):
    for j in range(M):
        c[i, j] = 0.0 
        for k in range(K):
            c[i, j] += a[i, k] + b[k, j]


***
conv(B, Cout, Cin, H, W, Kh, Kw, Sh, Sw):
for b in range(B):
    for co in range(Cout):
        for h in range(H):
            for w in range(W):
                c[b, co, h, w] = 0.0
                for ci in range(Cin):
                    for kh in range:
                        for kw in range:
                            c[b, co, h, w] += c[b, ci, h + kh, w + kw] * w[co, ci, kh, kw]

***
activation(N):
for i in range(N):
    c[i] = activate(c[i])

---------------------------------
Normalized Computation Definition

for out_axes in grid(out_extents):
    output[out_axes] = initial_value
    for in_axes in grid(in_extents):
        outputs[out_axes] = expr(outputs[out_axes], 
                                 input_1[linear_combination_1(in_axes, out_axes)], 
                                 ..., 
                                 input_q[linear_combination_q(in_axes, out_axes)])

--------------------------------
Target Hierarchical Architecture

CPUs:
cpu             cores           single core
global memory   L2 cache        L1 cache, registers

NVIDIA GPUs:
grid            block           thread
global memory   shared memory   registers

"""
