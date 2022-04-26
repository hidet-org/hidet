from hidet.ir.func import IRModule
from .utils import Task, Operator, Tensor, TensorInput, compute, reduce, input_like


class MatmulTask(Task):
    def __init__(self, A: TensorInput, B: TensorInput):
        BS, M, K = A.const_shape()
        BS, K, N = B.const_shape()
        C = compute(
            name='C',
            shape=[BS, M, N],
            fcompute=lambda b, i, j: reduce([K], lambda k: A[b, i, k] * B[b, k, j], 'sum'),
            scope='global'
        )
        super().__init__(
            name='matmul',
            inputs=[A, B],
            outputs=[C]
        )


class MatmulOp(Operator):
    def __init__(self, a: Tensor, b: Tensor):
        if not (len(a.shape) == len(b.shape) == 3 and a.shape[0] == b.shape[0] and a.shape[2] == b.shape[1]):
            raise ValueError('Matrix multiplication expect tensor A and B with shape [B, M, K] and [B, K, N]' +
                             ', got {} and {}'.format(a.shape, b.shape))
        task = MatmulTask(input_like(a, 'A'), input_like(b, 'B'))
        super().__init__([a, b], task)


def matmul(a: Tensor, b: Tensor) -> Tensor:
    if not (len(a.shape) == 2 and len(b.shape) == 2 and a.shape[1] == b.shape[0]):
        raise ValueError('Can not do matrix multiplication on {} and {}'.format(a.shape, b.shape))
    a = a.unsqueeze(0)
    b = b.unsqueeze(0)
    c = batched_matmul(a, b)
    return c.squeeze(0)


def batched_matmul(a: Tensor, b: Tensor) -> Tensor:
    return MatmulOp(a, b).get_output(0)
