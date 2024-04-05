import math
import torch
import numpy as np

import hidet
from hidet.cuda.cublas import cublasComputeType
from hidet.utils.benchmark import do_bench
from hidet import ops


def benchmark_cublas_batched_gemm(bs, m, n, k, dtype, compute_type):
    a, b, c = [], [], []
    for i in range(bs):
        a.append(hidet.randn((m, k), device='cuda', dtype=dtype) / math.sqrt(k))
        b.append(hidet.randn((k, n), device='cuda', dtype=dtype) / math.sqrt(k))
        c.append(hidet.empty((m, n), device='cuda', dtype=dtype))

    latencies = do_bench(
        lambda: hidet.cuda.cublas.batched_gemm(
            bs, m, n, k, a[0].dtype, b[0].dtype, c[0].dtype, a, b, c, False, False, compute_type
        ),
        warmup=10,
        rep=100,
    )

    print(f"cublas_batched_gemm Results for Configuration: dtype = {dtype}, input shape = {[bs, m, n, k]}, ")
    print("Median Latency Is: " + str(latencies[1]) + " milliseconds")
    print("-------------------------------------------------")


def benchmark_cublas_strided_gemm(bs, m, n, k, dtype, compute_type):
    a = hidet.randn((bs, m, k), device='cuda', dtype=dtype) / math.sqrt(k)
    b = hidet.randn((bs, k, n), device='cuda', dtype=dtype) / math.sqrt(k)
    c = hidet.empty((bs, m, n), device='cuda', dtype=dtype)

    latencies = do_bench(
        lambda: hidet.cuda.cublas.strided_gemm(
            bs, m, n, k, a.dtype, b.dtype, c.dtype, a, b, c, m * k, k * n, m * n, False, False, compute_type
        ),
        warmup=10,
        rep=100,
    )

    print(f"cublas_strided_gemm Results for Configuration: dtype = {dtype}, input shape = {[bs, m, n, k]}, ")
    print("Median Latency Is: " + str(latencies[1]) + " milliseconds")
    print("-------------------------------------------------")


def benchmark_torch_batched_matmul(bs, m, n, k, dtype, compute_type):
    a = torch.from_numpy(np.array(np.random.randn(bs, m, k)).astype(dtype)).cuda()
    b = torch.from_numpy(np.array(np.random.randn(bs, k, n)).astype(dtype)).cuda()

    latencies = do_bench(lambda: a @ b, warmup=10, rep=100)

    print(f"torch_batched_matmul Results for Configuration: dtype = {dtype}, input shape = {[bs, m, n, k]}, ")
    print("Median Latency Is: " + str(latencies[1]) + " milliseconds")
    print("-------------------------------------------------")


def benchmark_hidet_batched_matmul(bs, m, n, k, dtype, compute_type):
    a = hidet.symbol((bs, m, k), device='cuda', dtype=dtype)
    b = hidet.symbol((bs, k, n), device='cuda', dtype=dtype)
    c = ops.matmul(a, b)
    hidet.option.search_space(2)
    graph = hidet.trace_from(c, inputs=[a, b])
    graph = hidet.graph.optimize(graph)
    graph = graph.cuda_graph()

    latencies = do_bench(lambda: graph.run_async(), warmup=10, rep=100)

    print(f"hidet_batched_matmul Results for Configuration: dtype = {dtype}, input shape = {[bs, m, n, k]}, ")
    print("Median Latency Is: " + str(latencies[1]) + " milliseconds")
    print("-------------------------------------------------")


if __name__ == '__main__':
    sizes = [
        # # Group 1
        [1, 512, 512, 512],
        [2, 512, 512, 512],
        [4, 512, 512, 512],
        [8, 512, 512, 512],
        # Group 2
        [1, 1024, 1024, 2048],
        [2, 1024, 1024, 2048],
        [4, 1024, 1024, 2048],
        [8, 1024, 1024, 2048],
    ]
    dtypes = [['float32', cublasComputeType.CUBLAS_COMPUTE_32F], ['float16', cublasComputeType.CUBLAS_COMPUTE_16F]]

    for data_type in dtypes:
        for size in sizes:
            # benchmark_cublas_batched_gemm(*(size + data_type))
            benchmark_cublas_strided_gemm(*(size + data_type))
            # benchmark_torch_batched_matmul(*(size + data_type))
            # benchmark_hidet_batched_matmul(*(size + data_type))
