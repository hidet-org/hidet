from typing import List, Type, Optional
from hidet.graph.ir import Operator, Tensor
from hidet.graph.transforms.base import PassContext
from hidet.graph.ops.definitions.matmul.matmul import matmul
from hidet.graph.ops.definitions.matmul.parallel_k_matmul import parallel_k_batched_matmul, parallel_k_nparts, parallel_k_batched_matmul_search
from hidet.graph.ops.definitions import MatmulOp
from .base import ResolveRule


def op_use_parallel_k(op: MatmulOp) -> bool:
    a, b = op.inputs
    batch_size, m_size, k_size = a.shape
    n_size = b.shape[2]
    return parallel_k_nparts(batch_size, m_size, n_size, k_size) != 1


class MatmulResolveRule(ResolveRule):
    def op_cls(self) -> Type[Operator]:
        return MatmulOp

    def resolve(self, op: Operator) -> Optional[List[Tensor]]:
        assert isinstance(op, MatmulOp)
        a, b = op.inputs
        if op.attrs['algo'] == 'default':
            parallel_k = PassContext.current().configs['parallel_k']
            if parallel_k == 'disabled':
                return [matmul(a, b, 'direct', mma=op.attrs['mma'])]
            elif parallel_k == 'default':
                if op_use_parallel_k(op):
                    return [parallel_k_batched_matmul(a, b, mma=op.attrs['mma'])]
                else:
                    return [matmul(a, b, 'direct', mma=op.attrs['mma'])]
            elif parallel_k == 'search':
                return [parallel_k_batched_matmul_search(a, b, mma=op.attrs['mma'])]
            elif isinstance(parallel_k, int):
                return [parallel_k_batched_matmul(a, b, mma=op.attrs['mma'], nparts=parallel_k)]
            else:
                raise ValueError('Can not recognize parallel_k config: {}'.format(parallel_k))

        return None
