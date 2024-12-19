import argparse
import hidet

from hidet.testing.torch_utils import bench_model
from hidet.testing.utils import init_hidet


def bench_matmul_f16(params: str, *args, **kwargs) -> float:
    a_shape, b_shape = params.split(',')
    a_shape = [int(s) for s in a_shape.split('x')]
    b_shape = [int(s) for s in b_shape.split('x')]
    a = hidet.symbol(a_shape, dtype='float16', device='cuda')
    b = hidet.symbol(b_shape, dtype='float16', device='cuda')
    c = hidet.ops.matmul(a, b)
    g = hidet.trace_from(c, inputs=[a, b])
    g = hidet.graph.optimize(g)
    g = g.cuda_graph()
    return bench_model(lambda: g.run_async(), [])


def bench_batch_matmul(params: str, *args, **kwargs) -> float:
    # Default to benchmarking f32 for now, though this op can run other dtypes
    a_shape, b_shape = params.split(',')
    a_shape = [int(s) for s in a_shape.split('x')]
    b_shape = [int(s) for s in b_shape.split('x')]
    a = hidet.symbol(a_shape, dtype='float32', device='cuda')
    b = hidet.symbol(b_shape, dtype='float32', device='cuda')
    c = hidet.ops.matmul(a, b)
    g = hidet.trace_from(c, inputs=[a, b])
    g = hidet.graph.optimize(g)
    g = g.cuda_graph()
    return bench_model(lambda: g.run_async(), [])


def bench_conv2d(params: str, *args, **kwargs) -> float:
    x_shape, w_shape = params.split(',')
    x_shape = [int(s) for s in x_shape.split('x')]
    w_shape = [int(s) for s in w_shape.split('x')]
    x = hidet.symbol(x_shape, dtype='float32', device='cuda')
    w = hidet.randn(w_shape, dtype='float32', device='cuda')
    o = hidet.ops.conv2d(x, w)
    g = hidet.trace_from(o, inputs=[x, w])
    g = hidet.graph.optimize(g)
    g = g.cuda_graph()
    return bench_model(lambda: g.run_async(), [])


def bench_transpose2d(params: str, *args, **kwargs) -> float:
    dtype = args[0] if args else "float32"
    x_shape = params
    x_shape = [int(s) for s in x_shape.split('x')]
    x = hidet.symbol(x_shape, dtype=dtype, device='cuda')
    o = hidet.ops.transpose(x)
    g = hidet.trace_from(o, inputs=[x])
    g = hidet.graph.optimize(g)
    g = g.cuda_graph()
    return bench_model(lambda: g.run_async(), [])


def bench_conv2d_gemm_f16(params: str, *args, **kwargs) -> float:
    x_shape, w_shape = params.split(',')
    x_shape = [int(s) for s in x_shape.split('x')]
    w_shape = [int(s) for s in w_shape.split('x')]
    x = hidet.symbol(x_shape, dtype='float16', device='cuda')
    w = hidet.randn(w_shape, dtype='float16', device='cuda')
    o = hidet.ops.conv2d(x, w)
    g = hidet.trace_from(o, inputs=[x, w])
    g = hidet.graph.optimize(g)
    g = g.cuda_graph()
    return bench_model(lambda: g.run_async(), [])


def bench_attn(params: str, *args, **kwargs) -> float:
    bs, seqlen, nhead, hdim = [int(s) for s in params.split('x')]
    q_shape = [bs, nhead, seqlen, hdim]
    k_shape = [bs, nhead, hdim, seqlen]
    v_shape = [bs, nhead, seqlen, hdim]
    q = hidet.symbol(q_shape, dtype='float16', device='cuda')
    k = hidet.symbol(k_shape, dtype='float16', device='cuda')
    v = hidet.symbol(v_shape, dtype='float16', device='cuda')
    o = hidet.ops.attention(q, k, v)
    g = hidet.trace_from(o, inputs=[q, k, v])
    g = hidet.graph.optimize(g)
    g = g.cuda_graph()
    return bench_model(lambda: g.run_async(), [])


def bench_attn_mask_add(params: str, *args, **kwargs) -> float:
    bs, seqlen, nhead, hdim = [int(s) for s in params.split('x')]
    q_shape = [bs, nhead, seqlen, hdim]
    k_shape = [bs, nhead, hdim, seqlen]
    v_shape = [bs, nhead, seqlen, hdim]
    mask_shape = [1, 1, 1, seqlen]
    q = hidet.symbol(q_shape, dtype='float16', device='cuda')
    k = hidet.symbol(k_shape, dtype='float16', device='cuda')
    v = hidet.symbol(v_shape, dtype='float16', device='cuda')
    mask = hidet.randn(mask_shape, dtype='float16', device='cuda')
    o = hidet.ops.attention(q, k, v, mask=mask)
    g = hidet.trace_from(o, inputs=[q, k, v, mask])
    g = hidet.graph.optimize(g)
    g = g.cuda_graph()
    return bench_model(lambda: g.run_async(), [])


def bench_reduce(params: str, *args, **kwargs) -> float:
    x_shape, axis = params.split(',', maxsplit=1)
    start = axis.find('axis=[') + len('axis=[')
    end = axis.find(']', start)
    axis = [int(s) for s in axis[start:end].split(',')]
    x_shape = [int(s) for s in x_shape.split('x')]
    x = hidet.symbol(x_shape, dtype='float16', device='cuda')
    o = hidet.ops.sum(x, dims=axis)
    g = hidet.trace_from(o, inputs=[x])
    g = hidet.graph.optimize(g)
    g = g.cuda_graph()
    return bench_model(lambda: g.run_async(), [])


bench_func_map = {
    'matmul_f16': bench_matmul_f16,
    'batch_matmul': bench_batch_matmul,
    'conv2d': bench_conv2d,
    'transpose2d': bench_transpose2d,
    'conv2d_gemm_f16': bench_conv2d_gemm_f16,
    'attn': bench_attn,
    'attn_mask_add': bench_attn_mask_add,
    'reduce': bench_reduce,
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Benchmark Operators')
    parser.add_argument('operator', type=str, help='Specify operator. E.g., matmul_f16')
    parser.add_argument(
        '--params', type=str, help='Specify Input Parameters. Different operators have different formats.'
    )
    parser.add_argument('--dtype', type=str, default='float16', help='Specify precision. E.g., float32')
    parser.add_argument('--backend', type=str, default='hidet', help='Unused')
    parser.add_argument('--mode', type=str, default='max-autotune', help='Unused')

    args = parser.parse_args()
    assert args.backend == 'hidet'

    operator, dtype = args.operator, args.dtype
    params = args.params
    if operator in bench_func_map:
        bench_func = bench_func_map[operator]
    else:
        raise ValueError(f'Benchmark function for operator {operator} not implemented')

    init_hidet()

    with hidet.graph.PassContext() as ctx:
        ctx.set_reduce_precision(dtype)
        latency = bench_func(params, dtype)
    print(latency)
