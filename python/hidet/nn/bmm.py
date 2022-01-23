from hidet.ir.type import tensor_type
from hidet.ir.expr import scalar_var, var
from hidet.ir.dialects.compute import tensor_input, compute, reduce_sum
from hidet.ir.task import Task, Grid


def bmm(B: int, N: int, M: int, K: int) -> Task:
    A = tensor_input('A', 'float32', [B, N, K])
    B = tensor_input('B', 'float32', [B, K, M])
    k = var()
    C = compute('C', [B, N, M], lambda b, i, j: reduce_sum(A[b, i, k] * B[b, k, j], axis=k, shape=[K]))
    return Task(
        name='matmul.grid',
        computation=C,
        params=[A, B, C],
        params_type=[tensor_type('global', 'float32', [N, K], layout=[K, 1]),
                     tensor_type('global', 'float32', [K, M], layout=[M, 1]),
                     tensor_type('global', 'float32', [N, M], layout=[M, 1])],
        worker=Grid()
    )


def generic_bmm() -> Task:
    B = scalar_var('B', 'int32')
    N = scalar_var('N', 'int32')
    M = scalar_var('M', 'int32')
    K = scalar_var('K', 'int32')
    A = tensor_input('A', 'float32', [B, N, K])
    B = tensor_input('B', 'float32', [B, K, M])
    k = Axis(K)
    C = compute('C', [B, N, M], lambda b, i, j: reduce_sum(A[b, i, k] * B[b, k, j], axis=k))
    return Task(
        name='matmul.grid',
        computation=C,
        params=[A, B, C],
        params_type=[tensor_type('global', 'float32', [N, K], layout=[K, 1]),
                     tensor_type('global', 'float32', [K, M], layout=[M, 1]),
                     tensor_type('global', 'float32', [N, M], layout=[M, 1])],
        worker=Grid()
    )

