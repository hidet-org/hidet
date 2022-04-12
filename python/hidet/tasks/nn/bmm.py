from hidet.ir.type import tensor_type
from hidet.ir.expr import scalar_var, var
from hidet.ir.dialects.compute import tensor_input, compute, reduce_sum
from hidet.ir.task import Task, Grid


def bmm(B: int, N: int, M: int, K: int) -> Task:
    A = tensor_input('A', 'float32', [B, N, K], 'global')
    B = tensor_input('B', 'float32', [B, K, M], 'global')
    k = var()
    C = compute('C', [B, N, M], lambda b, i, j: reduce_sum(A[b, i, k] * B[b, k, j], axes=k, shape=[K]), scope='global')
    return Task(
        name='matmul.grid',
        computation=C,
        params=[A, B, C],
        worker=Grid()
    )
