import numpy as np
import torch

import hidet
from hidet.cuda.cudnn import cudnnDataType
from hidet.utils.benchmark import do_bench


def benchmark_cudnn_conv2d(dtype_str, compute_type, n, c, h, w, k, p, q, r, s, padding, stride, dilations):
    if dtype_str == "float32":
        dtype = hidet.float32
    elif dtype_str == "float64":
        dtype = hidet.float64
    else:
        raise Exception("Unsupported DataType")

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
        rep=100,
    )

    print(
        f"CuDNN Results for Configuration: dtype = {dtype_str}, input shape = {[n,c,h,w]}, "
        f"weight shape = {[k,c,r,s]}, padding = {padding}, stride = {stride}, dilations = {dilations}:"
    )
    print("20th Percentile Latency Is: " + str(latencies[0]) + " milliseconds")
    print("50th Percentile Latency Is: " + str(latencies[1]) + " milliseconds")
    print("80th Percentile Latency Is: " + str(latencies[2]) + " milliseconds")
    print("-------------------------------------------------")


def benchmark_torch_conv2d(dtype_str, compute_type, n, c, h, w, k, p, q, r, s, padding, stride, dilations):
    if dtype_str == "float32":
        dtype = np.float32
    elif dtype_str == "float64":
        dtype = np.float64
    else:
        raise Exception("Unsupported DataType")

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
        f"PyTorch Results for Configuration: dtype = {dtype_str}, input shape = {[n,c,h,w]}, "
        f"weight shape = {[k,c,r,s]}, padding = {padding}, stride = {stride}, dilations = {dilations}:"
    )
    print("20th Percentile Latency Is: " + str(latencies[0]) + " milliseconds")
    print("50th Percentile Latency Is: " + str(latencies[1]) + " milliseconds")
    print("80th Percentile Latency Is: " + str(latencies[2]) + " milliseconds")
    print("-------------------------------------------------")


if __name__ == '__main__':
    sizes = [
        [1, 3, 32, 32, 12, 30, 30, 3, 3, [0, 0], [1, 1], [1, 1]],
        [2, 3, 224, 224, 16, 109, 109, 7, 7, [0, 0], [2, 2], [1, 1]],
    ]
    dtypes = [['float32', cudnnDataType.CUDNN_DATA_FLOAT], ['float64', cudnnDataType.CUDNN_DATA_DOUBLE]]

    for data_type in dtypes:
        for size in sizes:
            benchmark_cudnn_conv2d(*(data_type + size))
            benchmark_torch_conv2d(*(data_type + size))
