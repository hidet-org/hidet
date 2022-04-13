from typing import Union, Sequence
from ..common import Task, Operator, Tensor, DataLayout, TensorInput, Grid, tensor_input, compute, reduce, inline_compute, tensor_type, input_like


def matmul_task(A: TensorInput, B: TensorInput) -> Task:
    M, K = A.const_shape()
    K, N = B.const_shape()
    C = compute('C', [M, N], lambda i, j: reduce([K], lambda k: A[i, k] * B[k, j], 'sum'), scope='global')
    return Task(
        name='matmul',
        computation=C,
        params=[A, B, C],
        worker=Grid()
    )


def batched_matmul_task(A: TensorInput, B: TensorInput) -> Task:
    B, M, K = A.const_shape()
    B, K, N = B.const_shape()
    C = compute('C', [B, M, N], lambda b, i, j: reduce([K], lambda k: A[b, i, k] * B[b, k, j], 'sum'), scope='global')
    return Task(
        name='batched_matmul',
        computation=C,
        params=[A, B, C],
        worker=Grid()
    )


class MatmulOp(Operator):
    def __init__(self, a: Tensor, b: Tensor):
        if not (len(a.shape) == len(b.shape) == 2 and a.shape[1] == b.shape[0]):
            raise ValueError('Can not do matrix multiplication over matrices with shape {} and {}'.format(a.shape, b.shape))
        task = matmul_task(input_like(a, 'A'), input_like(b, 'B'))
        super().__init__([a, b], task)


class BatchedMatmulOp(Operator):
    def __init__(self, a: Tensor, b: Tensor):
        if not (len(a.shape) == len(b.shape) == 3 and a.shape[0] == b.shape[0] and a.shape[2] == b.shape[1]):
            raise ValueError('Batched matrix multiplication expect tensor A and B with shape [B, M, K] and [B, K, N]' +
                             ', got {} and {}'.format(a.shape, b.shape))
        task = batched_matmul_task(input_like(a, 'A'), input_like(b, 'B'))
        super().__init__([a, b], task)


def matmul(a: Tensor, b: Tensor) -> Tensor:
    return MatmulOp(a, b).get_output(0)


def batched_matmul(a: Tensor, b: Tensor) -> Tensor:
    return BatchedMatmulOp(a, b).get_output(0)
