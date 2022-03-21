from hidet.ir.type import tensor_type
from hidet.ir.expr import scalar_var, var
from hidet.ir.dialects.compute import tensor_input, compute, reduce_sum
from hidet.ir.task import Task, Grid, ThreadBlock


def matmul(M: int, N: int, K: int, worker=Grid()) -> Task:
    A = tensor_input('A', 'float32', [M, K])
    B = tensor_input('B', 'float32', [K, N])
    k = var('k')
    C = compute('C', [M, N], lambda i, j: reduce_sum(A[i, k] * B[k, j], axes=k, shape=[K]))
    return Task(
        name='matmul',
        computation=C,
        params=[A, B, C],
        params_type=[tensor_type('global', 'float32', [M, K], layout=[K, 1]),
                     tensor_type('global', 'float32', [K, N], layout=[N, 1]),
                     tensor_type('global', 'float32', [M, N], layout=[N, 1])],
        worker=worker
    )


def generic_matmul() -> Task:
    M = scalar_var('M', 'int32')
    N = scalar_var('N', 'int32')
    K = scalar_var('K', 'int32')
    A = tensor_input('A', 'float32', [M, K])
    B = tensor_input('B', 'float32', [K, N])
    k = var('k')
    C = compute('C', [M, N], lambda i, j: reduce_sum(A[i, k] * B[k, j], axes=k, shape=[K]))
    return Task(
        name='matmul',
        computation=C,
        params=[A, B, C],
        params_type=[tensor_type('global', 'float32', [M, K], layout=[K, 1]),
                     tensor_type('global', 'float32', [K, N], layout=[N, 1]),
                     tensor_type('global', 'float32', [M, N], layout=[N, 1])],
        worker=Grid()
    )

