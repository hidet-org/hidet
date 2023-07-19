import os
import json
import numpy as np
import torch
import hidet
import argparse
from result_entry import ResultEntry, ResultGroup, load_regression_data

device_name = str(hidet.cuda.properties().name, 'UTF-8')

def bench_matmul(m, n, k, dtype):
    hidet.option.search_space(2)
    a = hidet.symbol([m, k], dtype=dtype, device='cuda')
    b = hidet.symbol([k, n], dtype=dtype, device='cuda')
    c = hidet.ops.matmul(a, b)
    g = hidet.trace_from(c, [a, b])
    g = hidet.graph.optimize(g)
    return g.latency()

def bench_fmha(sq, skv, d):
    hidet.option.search_space(2)
    q = hidet.symbol([sq, d], dtype='float16', device='cuda')
    k = hidet.symbol([d, skv], dtype='float16', device='cuda')
    v = hidet.symbol([skv, d], dtype='float16', device='cuda')
    o = hidet.ops.attention(q, k, v)
    g = hidet.trace_from(o, [q, k, v])
    g = hidet.graph.optimize(g)
    return g.latency()

def matmul_regression() -> ResultGroup:
    regression_data = load_regression_data()
    result_group = ResultGroup(name='Matrix Multiply Regression', device_name=device_name)
    matmul_data = regression_data[device_name]['matmul_shapes']
    for shape, perf in matmul_data.items():
        for dtype, ref_latency in perf.items():
            (m, n, k) = [int(s.strip()) for s in shape.split(',')]
            latency = bench_matmul(m, n, k, dtype)
            if not np.allclose(latency, ref_latency, rtol=0.05):
                # ToDo: deal with slowdown/speedup
                pass
            result_group.add_entry(ResultEntry(shape, dtype, latency))
    return result_group


def fmha_regression() -> ResultGroup:
    regression_data = load_regression_data()
    result_group = ResultGroup(name='Fused Attention Regression', device_name=device_name)
    fmha_data = regression_data[device_name]['fmha_shapes']
    for shape, perf in fmha_data.items():
        for dtype, ref_latency in perf.items():
            (sq, skv, d) = [int(s.strip()) for s in shape.split(',')]
            latency = bench_fmha(sq, skv, d)
            if not np.allclose(latency, ref_latency, rtol=0.05):
                # ToDo: deal with slowdown/speedup
                pass
            result_group.add_entry(ResultEntry(shape, dtype, latency))
    return result_group

def conv2d_regression() -> ResultGroup:
    # ToDo
    return None

def reduce_regression() -> ResultGroup:
    # ToDo
    return None


def op_performance_regression(report_file):
    result_groups = []
    result_groups.append(matmul_regression())
    result_groups.append(fmha_regression())
    result_groups.append(conv2d_regression())
    result_groups.append(reduce_regression())
    with open(report_file, 'w') as f:
        f.write("---------------- Operator Performance Regression -----------------\n")
        for result_group in result_groups:
            if result_group is not None:
                f.write(str(result_group))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Operator Performance Regression')
    parser.add_argument(
        '--report',
        type=str,
        default='./report_op_performance.txt',
        help='Specify report output path'
    )
    args = parser.parse_args()
    op_performance_regression(args.report)