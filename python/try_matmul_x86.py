import numpy as np
import pytest

import hidet
from hidet.graph.ops import matmul_x86
from hidet.testing import check_binary
from hidet.option import debug_cache_tuning

import tvm
from tvm import te, auto_scheduler

@auto_scheduler.register_workload
def matmul_ansor(M, K, N, dtype):
    A = te.placeholder((M, K), name="A", dtype=dtype)
    B = te.placeholder((K, N), name="B", dtype=dtype)
    # C = te.placeholder((M, N), name="C", dtype=dtype)

    k = te.reduce_axis((0, K), name="k")
    rst = te.compute(
        (M, N),
        lambda i, j: te.sum(A[i, k] * B[k, j], axis=k),
        name="matmul_ansor",
        attrs={"layout_free_placeholders": [B]},        # Enable automatic layout transform for B TODO: What is this?
    )

    return [A, B, rst]

target = tvm.target.Target("llvm -mcpu=core-avx2")


debug_cache_tuning(True)
hidet.option.search_space(0)
# hidet.option.cache_dir("./wtfisthis")
hidet.option.parallel_build(True)
# for m, k, n in [(18, 32, 96), (24, 64, 256), (24, 64, 512), (192, 64, 128), (192, 128, 128), (192, 256, 256), (784, 40, 120), (784, 120, 40), (480, 512, 16), (384, 384, 32), (784, 40, 120),
#                 (256, 256, 256), (384, 256, 256),
#                 (384, 384, 512), (512, 512, 512), (1369, 48, 256),
#                 (1024, 1024, 1024), (2048, 2048, 2048), (1024, 3072, 512), (512, 3072, 1024), (1369, 64, 288), (4096, 4096, 4096),
#                 (22500, 32, 27), (22201, 32, 288),
#                 (3136, 64, 64), (2500, 32, 27), (3329, 192, 720)]:
for m, n, k in [(1440, 1440, 1440)]:
    a = hidet.randn([m, k], device='cpu')
    b = hidet.randn([k, n], device='cpu')
    # c = matmul_x86(a, b)
    x1 = hidet.symbol_like(a)
    x2 = hidet.symbol_like(b)
    y = matmul_x86(x1, x2)
    graph: hidet.FlowGraph = hidet.trace_from(y, inputs=[x1, x2])
    opt_graph = hidet.graph.optimize(graph)
    compiled_func = opt_graph.nodes[0].task_func

    c = hidet.zeros([m, n], device='cpu')

    compiled_func(a, b, c)

    np.testing.assert_allclose(
        actual=c.numpy(),
        desired=a.numpy() @ b.numpy(),
        rtol=1e-3,
        atol=1e-3
    )
    hidet_latency = hidet.utils.benchmark_func(
        lambda: compiled_func(a, b, c), repeat=50
    )
    np_latency = hidet.utils.benchmark_func(
        lambda: a.numpy() @ b.numpy(), repeat=50
    )

    # ansor_task = tvm.auto_scheduler.SearchTask(func=matmul_ansor, args=(m, k, n, "float32"), target=target)
    # log_file = f"matmul_{m}x{k}x{k}.json"
    # tune_option = auto_scheduler.TuningOptions(
    #     num_measure_trials=1000,
    #     measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
    #     verbose=2,
    # )
    #
    # ansor_task.tune(tune_option)
    # sch, args = ansor_task.apply_best(log_file)
    #
    # with open(f"./matmul_TIR_{m}x{k}x{n}", 'w') as f:
    #     f.write(str(tvm.lower(sch, args, simple_mode=True)))
    # ansor_func = tvm.build(sch, args, target)
    # dev = tvm.cpu()
    # a_tvm = tvm.nd.array(a.numpy(), device=dev)
    # b_tvm = tvm.nd.array(b.numpy(), device=dev)
    # c_tvm = tvm.nd.empty((m, n), device=dev)
    #
    # ansor_func(a_tvm, b_tvm, c_tvm)
    #
    # np.testing.assert_allclose(
    #     actual=c_tvm.numpy(),
    #     desired=a_tvm.numpy() @ b_tvm.numpy(),
    #     rtol=1e-3,
    #     atol=1e-3
    # )
    #
    # ansor_latency = hidet.utils.benchmark_func(
    #     lambda: ansor_func(a_tvm, b_tvm, c_tvm), repeat=30
    # )

    with open(f"./perf_dynamic_arith-1440.txt", 'a+') as f:
        f.write(f'm={m}, k={k}, n={n}: hidet takes {hidet_latency:.2f} ms\n')
        f.write(f'm={m}, k={k}, n={n}: numpy takes {np_latency:.2f} ms\n')
        # f.write(f'm={m}, k={k}, n={n}: ansor takes {ansor_latency:.2f} ms\n')



