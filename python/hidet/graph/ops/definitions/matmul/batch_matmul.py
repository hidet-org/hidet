# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from hidet.ir.func import IRModule
from hidet.graph.ops.definitions.utils import Task, Operator, Tensor, TensorNode, compute, reduce, input_like


class BatchMatmulTask(Task):
    def __init__(self, a: TensorNode, b: TensorNode, mma: str = 'simt'):
        batch_size, m_size, k_size = a.const_shape()
        batch_size, k_size, n_size = b.const_shape()
        self.batch_size: int = batch_size
        self.m_size: int = m_size
        self.k_size: int = k_size
        self.n_size: int = n_size
        self.mma: str = mma
        c = compute(
            name='c',
            shape=[batch_size, m_size, n_size],
            fcompute=lambda r, i, j: reduce(
                shape=[k_size], fcompute=lambda k: a[r, i, k] * b[r, k, j], reduce_type='sum'
            ),
        )
        super().__init__(
            name='batch_matmul',
            inputs=[a, b],
            outputs=[c],
            attributes={'batch_size': batch_size, 'm_size': m_size, 'n_size': n_size, 'k_size': k_size, 'mma': mma},
        )

    def implement_cuda(self, workding_dir: str) -> IRModule:
        from hidet.graph.ops.schedules.cuda import matmul as matmul_schedule  # pylint: disable=import-outside-toplevel

        if self.mma == 'simt':
            return matmul_schedule.batched_matmul_cuda_schedule_simt(self, workding_dir)
        elif self.mma.startswith('wmma'):
            return matmul_schedule.batched_matmul_cuda_schedule_wmma(self, workding_dir)
        elif self.mma.startswith('mma'):
            return matmul_schedule.batched_matmul_cuda_schedule_mma(self, workding_dir)
        else:
            raise ValueError('Can not recognize mma type {}, candidates: {}'.format(self.mma, ['simt', 'wmma', 'mma']))


class BatchMatmulOp(Operator):
    def __init__(self, a: Tensor, b: Tensor, mma: str = 'simt'):
        if not (len(a.shape) == len(b.shape) == 3 and a.shape[0] == b.shape[0] and a.shape[2] == b.shape[1]):
            raise ValueError(
                'Matrix multiplication expect tensor A and B with shape [B, M, K] and [B, K, N]'
                + ', got {} and {}'.format(a.shape, b.shape)
            )
        task = BatchMatmulTask(input_like(a, 'a'), input_like(b, 'b'), mma)
        super().__init__(inputs=[a, b], task=task, attributes={'mma': mma})


def batch_matmul(a: Tensor, b: Tensor, mma: str = 'simt') -> Tensor:
    """Batched matrix multiplication.

    Parameters
    ----------
    a: Tensor
        The lhs operand with shape [batch_size, m_size, k_size].

    b: Tensor
        The rhs operand with shape [batch_size, k_size, n_size].

    mma: str
        The matrix-multiplication-accumulate (mma) in warp level:

        - 'simt':
           Use cuda core to do the warp-level mma (simt stands for single-instruction-multiple-threads).
        - 'wmma':
           Use wmma instruction.
        - 'mma':
           Use mma instruction.

        See also:
        https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions

    Returns
    -------
    c: Tensor
        The result tensor of matrix multiplication.
    """
    return BatchMatmulOp(a, b, mma).get_output(0)
