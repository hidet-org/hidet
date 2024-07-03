import numpy as np
import torch

import hidet
from hidet.cuda.cudnn import cudnnDataType
from hidet.utils.benchmark import do_bench
from hidet import ops


def benchmark_cudnn_conv2d(dtype, compute_type, n, c, h, w, k, p, q, r, s, padding, stride, dilations):
    # Uses ordinary cudnn.conv2d implemented with Graph-API
    tx = tw = ty = dtype
    pad_dim1, pad_dim2 = padding
    str_dim1, str_dim2 = stride
    dil_dim1, dil_dim2 = dilations

    tensor_x = hidet.randn((n, c, h, w), device='cuda', dtype=tx)
    tensor_w = hidet.randn((k, c, r, s), device='cuda', dtype=tw)
    tensor_y = hidet.empty((n, k, p, q), device='cuda', dtype=ty)

    latencies = do_bench(
        lambda: hidet.cuda.cudnn.conv2d(
            n,
            c,
            h,
            w,
            k,
            r,
            s,
            p,
            q,
            tensor_x,
            tensor_w,
            tensor_y,
            tx,
            tw,
            ty,
            compute_type,
            pad_dim1,
            pad_dim2,
            str_dim1,
            str_dim2,
            dil_dim1,
            dil_dim2,
        ),
        warmup=10,
        rep=1,
    )

    print(
        f"CuDNN Results for Configuration: dtype = {dtype}, input shape = {[n,c,h,w]}, "
        f"weight shape = {[k,c,r,s]}, padding = {padding}, stride = {stride}, dilations = {dilations}:"
    )
    print("Median Latency Is: " + str(latencies[1]) + " milliseconds")
    print("-------------------------------------------------")


def benchmark_cudnn_conv2d_gemm(dtype, compute_type, n, c, h, w, k, p, q, r, s, padding, stride, dilations):
    # Uses cudnn.conv2d_gemm implemented with Legacy-API
    tx = tw = ty = dtype
    pad_dim1, pad_dim2 = padding
    str_dim1, str_dim2 = stride
    dil_dim1, dil_dim2 = dilations

    tensor_x = hidet.randn((n, c, h, w), device='cuda', dtype=tx)
    tensor_w = hidet.randn((k, c, r, s), device='cuda', dtype=tw)
    tensor_y = hidet.empty((n, k, p, q), device='cuda', dtype=ty)

    latencies = do_bench(
        lambda: hidet.cuda.cudnn.conv2d_gemm(
            n,
            c,
            h,
            w,
            k,
            r,
            s,
            tensor_x,
            tensor_w,
            tensor_y,
            tx,
            tw,
            ty,
            compute_type,
            pad_dim1,
            pad_dim2,
            str_dim1,
            str_dim2,
            dil_dim1,
            dil_dim2,
        ),
        warmup=10,
        rep=100,
    )

    print(
        f"cudnn_gemm Results for Configuration: dtype = {dtype}, input shape = {[n,c,h,w]}, "
        f"weight shape = {[k,c,r,s]}, padding = {padding}, stride = {stride}, dilations = {dilations}:"
    )
    print("Median Latency Is: " + str(latencies[1]) + " milliseconds")
    print("-------------------------------------------------")


def benchmark_torch_conv2d(dtype, compute_type, n, c, h, w, k, p, q, r, s, padding, stride, dilations):
    # Native PyTorch Eager-mode Execution
    data = np.array(np.random.randn(n, c, h, w)).astype(dtype)
    weight = np.array(np.random.randn(k, c, r, s)).astype(dtype)

    data_torch, weight_torch = torch.from_numpy(data), torch.from_numpy(weight)
    data_torch = data_torch.cuda()
    weight_torch = weight_torch.cuda()

    latencies = do_bench(
        lambda: torch.nn.functional.conv2d(
            data_torch, weight_torch, bias=None, stride=stride, padding=padding, dilation=dilations, groups=1
        ),
        warmup=10,
        rep=100,
    )

    print(
        f"PyTorch Results for Configuration: dtype = {dtype}, input shape = {[n,c,h,w]}, "
        f"weight shape = {[k,c,r,s]}, padding = {padding}, stride = {stride}, dilations = {dilations}:"
    )
    print("Median Latency Is: " + str(latencies[1]) + " milliseconds")
    print("-------------------------------------------------")


def benchmark_hidet_conv2d(dtype, compute_type, n, c, h, w, k, p, q, r, s, padding, stride, dilations):
    # Uses optimized Hidet Graph implementation
    tx = tw = dtype
    pad_dim1, pad_dim2 = padding
    str_dim1, str_dim2 = stride
    dil_dim1, dil_dim2 = dilations

    hidet.option.search_space(2)
    tensor_x = hidet.symbol((n, c, h, w), device='cuda', dtype=tx)
    tensor_w = hidet.randn((k, c, r, s), device='cuda', dtype=tw)
    output = ops.conv2d(
        tensor_x, tensor_w, stride=(str_dim1, str_dim2), dilations=(dil_dim1, dil_dim2), padding=(pad_dim1, pad_dim2)
    )
    graph = hidet.trace_from(output, inputs=[tensor_x, tensor_w])
    graph = hidet.graph.optimize(graph)
    graph = graph.cuda_graph()

    latencies = do_bench(lambda: graph.run_async(), warmup=10, rep=100)

    print(
        f"Optimized Hidet Results for Configuration: dtype = {dtype}, input shape = {[n,c,h,w]}, "
        f"weight shape = {[k,c,r,s]}, padding = {padding}, stride = {stride}, dilations = {dilations}:"
    )
    print("Median Latency Is: " + str(latencies[1]) + " milliseconds")
    print("-------------------------------------------------")


if __name__ == '__main__':
    sizes = [
        # Group 1
        [1, 3, 224, 224, 64, 112, 112, 7, 7, [3, 3], [2, 2], [1, 1]],
        [2, 3, 224, 224, 64, 112, 112, 7, 7, [3, 3], [2, 2], [1, 1]],
        [4, 3, 224, 224, 64, 112, 112, 7, 7, [3, 3], [2, 2], [1, 1]],
        [8, 3, 224, 224, 64, 112, 112, 7, 7, [3, 3], [2, 2], [1, 1]],
        # Group 2
        [1, 64, 56, 56, 128, 56, 56, 1, 1, [0, 0], [1, 1], [1, 1]],
        [2, 64, 56, 56, 128, 56, 56, 1, 1, [0, 0], [1, 1], [1, 1]],
        [4, 64, 56, 56, 128, 56, 56, 1, 1, [0, 0], [1, 1], [1, 1]],
        [8, 64, 56, 56, 128, 56, 56, 1, 1, [0, 0], [1, 1], [1, 1]],
    ]
    dtypes = [['float32', cudnnDataType.CUDNN_DATA_FLOAT], ['float16', cudnnDataType.CUDNN_DATA_HALF]]

    for data_type in dtypes:
        for size in sizes:
            benchmark_cudnn_conv2d_gemm(*(data_type + size))
            benchmark_torch_conv2d(*(data_type + size))
            benchmark_hidet_conv2d(*(data_type + size))
