import numpy as np
import pytest

import hidet
from hidet.graph.ops import matmul_x86
from hidet.testing import check_binary
from hidet.option import debug_cache_tuning

debug_cache_tuning(True)
hidet.option.search_space(2)
for m, k, n in [(18, 32, 96), (24, 64, 256), (24, 64, 512), (192, 64, 128), (192, 128, 128), (192, 256, 256), (784, 40, 120), (784, 120, 40), (480, 512, 16), (384, 384, 32), (784, 40, 120),
                (256, 256, 256), (384, 256, 256),
                (384, 384, 512), (1369, 48, 256),
                (1024, 1024, 1024), (2048, 2048, 2048), (1024, 3072, 512), (512, 3072, 1024), (1369, 64, 288), (4096, 4096, 4096),
                (22500, 32, 27), (22201, 32, 288),
                (3136, 64, 64), (2500, 32, 27), (3329, 192, 720)]:
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
        lambda: compiled_func(a, b, c), repeat=30
    )
    np_latency = hidet.utils.benchmark_func(
        lambda: a.numpy() @ b.numpy(), repeat=30
    )

    print(f'm={m}, k={k}, n={n}: hidet takes {hidet_latency:.2f} ms')
    print(f'm={m}, k={k}, n={n}: numpy takes {np_latency:.2f} ms')
    print()


