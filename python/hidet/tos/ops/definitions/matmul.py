from hidet.ir.func import IRModule
from .utils import Task, Operator, Tensor, TensorInput, compute, reduce, input_like


class MatmulTask(Task):
    def __init__(self, a: TensorInput, b: TensorInput):
        batch_size, m_size, k_size = a.const_shape()
        batch_size, k_size, n_size = b.const_shape()
        self.batch_size: int = batch_size
        self.m_size: int = m_size
        self.k_size: int = k_size
        self.n_size: int = n_size
        c = compute(
            name='c',
            shape=[batch_size, m_size, n_size],
            fcompute=lambda r, i, j: reduce(
                shape=[k_size],
                fcompute=lambda k: a[r, i, k] * b[r, k, j],
                reduce_type='sum'
            ),
            scope='global'
        )
        super().__init__(
            name='matmul',
            inputs=[a, b],
            outputs=[c]
        )

    def implement_cuda(self) -> IRModule:
        from hidet.tos.ops.schedules.cuda import batched_matmul_cuda_schedule
        return batched_matmul_cuda_schedule(self, space_level=0)


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
