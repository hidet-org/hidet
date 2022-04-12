from hidet.ir.type import tensor_type
from hidet.ir.expr import scalar_var, var
from hidet.ir.dialects.compute import tensor_input, compute, reduce
from hidet.ir.task import Task, Grid, ThreadBlock
from hidet.ir.layout.data_layout import RowMajorLayout


def matmul(M: int, N: int, K: int, worker=Grid()) -> Task:
    A = tensor_input('A', 'float32', [M, K], 'global')
    B = tensor_input('B', 'float32', [K, N], 'global')
    C = compute('C', [M, N], lambda i, j: reduce([K], lambda k: A[i, k] * B[k, j], 'sum'), scope='global')
    return Task(
        name='matmul',
        computation=C,
        params=[A, B, C],
        worker=worker
    )


def generic_matmul() -> Task:
    M = scalar_var('M', 'int32')
    N = scalar_var('N', 'int32')
    K = scalar_var('K', 'int32')
    A = tensor_input('A', 'float32', [M, K], 'global')
    B = tensor_input('B', 'float32', [K, N], 'global')
    k = var('k')
    C = compute('C', [M, N], lambda i, j: reduce([K], lambda k: A[i, k] * B[k, j], 'sum'), scope='global')
    return Task(
        name='matmul',
        computation=C,
        params=[A, B, C],
        worker=Grid()
    )

