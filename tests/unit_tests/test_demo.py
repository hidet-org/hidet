import pytest
from hidet.ir.type import tensor_type
from hidet.ir.expr import var
from hidet.ir.task import Task, Grid
from hidet.ir.dialects.compute import tensor_input, reduce_sum, compute
from hidet.runtime.value import TensorValue
from hidet.implement import implement
from hidet.implement import random_resolve
from hidet.backend import build


def get_task(N=1024, M=1024, K=1024):
    k = var('k')

    A = tensor_input('A', 'float32', [N, K])
    B = tensor_input('B', 'float32', [K, M])
    C = compute('C', [N, M], lambda i, j: reduce_sum(A[i, k] * B[k, j], axes=k, shape=[K]))

    params_type = [
        tensor_type('global', 'float32', [N, K], [K, 1]),
        tensor_type('global', 'float32', [K, M], [M, 1]),
        tensor_type('global', 'float32', [N, M], [M, 1])
    ]
    task = Task('gemm', C, [A, B, C], params_type, Grid())
    return task


def test_demo():
    N, M, K = 2, 2, 2
    task = get_task(N, M, K)
    ir_module = implement(task)
    ir_module = random_resolve(ir_module)
    module = build(ir_module, output_dir='./outs')

    A = TensorValue.randn([N, K], 'float32', 'global', seed=1)
    B = TensorValue.randn([K, M], 'float32', 'global', seed=3)
    C = TensorValue.empty([N, M], 'float32', 'global')
    module['gemm'](A, B, C)


if __name__ == '__main__':
    pytest.main(__file__)
