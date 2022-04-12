import pytest
from hidet.ir.type import tensor_type
from hidet.ir.expr import var
from hidet.ir.task import Task, Grid
from hidet.ir.dialects.compute import tensor_input, compute, reduce
from hidet.implement import implement
from hidet.implement import random_resolve
from hidet.backend import build
from hidet.tos.tensor import from_numpy, randn, empty


def get_task(N=1024, M=1024, K=1024):
    k = var('k')

    A = tensor_input('A', 'float32', [N, K], scope='global')
    B = tensor_input('B', 'float32', [K, M], scope='global')
    C = compute('C', [N, M], lambda i, j: reduce([K], lambda k: A[i, k] * B[k, j], 'sum'), scope='global')

    task = Task('gemm', C, [A, B, C], Grid())
    return task


def test_demo():
    N, M, K = 2, 2, 2
    task = get_task(N, M, K)
    ir_module = implement(task)
    ir_module = random_resolve(ir_module)
    module = build(ir_module, output_dir='./outs')

    A = randn([N, K], 'float32', device='cuda')
    B = randn([K, M], 'float32', device='cuda')
    C = empty([N, M], 'float32', device='cuda')
    module['gemm'](A, B, C)


if __name__ == '__main__':
    pytest.main(__file__)
