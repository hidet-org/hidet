from typing import List, Type, Optional
from functools import lru_cache

import hidet
from hidet.graph.ir import Operator, Tensor
from hidet.graph.transforms import ResolveRule, register_resolve_rule
from hidet.utils.py import gcd, factorize
from hidet.ffi.cuda_api import cuda

from .matmul import MatmulOp
from .batch_matmul import batch_matmul


def parallel_k_heuristic_nparts(batch_size, m_size, n_size, k_size) -> int:
    estimate_thread_blocks = batch_size * ((m_size + 63) // 64) * ((n_size + 63) // 64)
    num_multi_processors = cuda.device_property(cuda.PropertyMultiProcessorCount)
    # we hope to run multiple waves of thread blocks (e.g., 5)
    if estimate_thread_blocks * 8 <= num_multi_processors * 5:
        nparts = 8
    elif estimate_thread_blocks * 4 <= num_multi_processors * 5:
        nparts = 4
    elif estimate_thread_blocks * 2 <= num_multi_processors * 5:
        nparts = 2
    else:
        nparts = 1
    return gcd(nparts, k_size)


@lru_cache(maxsize=1024)
def parallel_k_search_nparts(dtype: str, mma: str, batch_size, m_size, n_size, k_size) -> int:
    import hidet
    nparts_candidates = [nparts for nparts in factorize(k_size) if nparts <= 16]
    best_nparts = None
    best_nparts_latency = 1e9
    latencies = []

    print('search parallel k factor for [{} x {} x {} x {}] among {}'.format(batch_size, m_size, n_size, k_size, nparts_candidates))
    for nparts in nparts_candidates:
        a = hidet.symbol([batch_size, m_size, k_size], dtype=dtype, device='cuda')
        b = hidet.symbol([batch_size, k_size, n_size], dtype=dtype, device='cuda')
        aa = a.reshape([batch_size, m_size, nparts, k_size // nparts]).rearrange([[0, 2], [1], [3]])  # [batch_size * nparts, m_size, k_size // nparts]
        bb = b.reshape([batch_size, nparts, k_size // nparts, n_size]).rearrange([[0, 1], [2], [3]])  # [batch_size * nparts, k_size // nparts, n_size]
        cc = batch_matmul(aa, bb, mma=mma)
        c = cc.reshape([batch_size, nparts, m_size, n_size]).sum(1)

        graph: hidet.FlowGraph = hidet.trace_from(c, [a, b])
        with hidet.graph.PassContext() as ctx:
            graph: hidet.FlowGraph = hidet.graph.transforms.fuse_operator_pass()(graph)

        latency: float = graph.latency()
        latencies.append(latency)
        if latency < best_nparts_latency:
            best_nparts = nparts
            best_nparts_latency = latency
    print('got parallel k factor {} with latency {:.3f} ms ({})'.format(
        best_nparts, best_nparts_latency,
        ', '.join(['{:.3f}'.format(v) for v in latencies])))
    return best_nparts


@register_resolve_rule
class MatmulResolveRule(ResolveRule):
    """
    Resolve a generic matrix multiplication operator to a batched matrix multiplication operator.

    The generic matrix multiplication operator has the same semantics as numpy.matmul that accepts
    variable dimensions of inputs.

    On ther other hand, the batched matrix multiplication operator accepts inputs with shape:
    [batch_size, m_size, k_size] x [batch_size, k_size, n_size]

    This resolve rule also parallelize k dimension when possible, and determine the mma instruction.
    """
    def op_cls(self) -> Type[Operator]:
        return MatmulOp

    def run_batch_matmul(self, a: Tensor, b: Tensor) -> Tensor:
        parallel_k = self.get_config('parallel_k', default='default')   # 'default', 'search', 2, 4, ...
        mma = self.get_config('mma', default='simt')    # 'simt', 'mma', 'wmma'

        batch_size, m_size, n_size, k_size = a.shape[0], a.shape[1], b.shape[2], a.shape[2]
        if parallel_k == 'default':
            nparts = parallel_k_heuristic_nparts(batch_size, m_size, n_size, k_size)
        elif parallel_k == 'search':
            nparts = parallel_k_search_nparts(a.dtype, mma, batch_size, m_size, n_size, k_size)
        elif parallel_k == 'disabled':
            nparts = 1
        elif isinstance(parallel_k, int):
            nparts = gcd(parallel_k, k_size)
        else:
            raise ValueError(f'invalid parallel_k: {parallel_k}')

        if nparts == 1:
            c = batch_matmul(a, b, mma=mma)
        else:
            aa = a.reshape([batch_size, m_size, nparts, k_size // nparts]).rearrange([[0, 2], [1], [3]])  # [batch_size * nparts, m_size, k_size // nparts]
            bb = b.reshape([batch_size, nparts, k_size // nparts, n_size]).rearrange([[0, 1], [2], [3]])  # [batch_size * nparts, k_size // nparts, n_size]
            c = batch_matmul(aa, bb, mma=mma).reshape([batch_size, nparts, m_size, n_size]).sum(1)
        return c

    def resolve(self, op: Operator) -> Optional[List[Tensor]]:
        assert isinstance(op, MatmulOp)
        a: Tensor = op.inputs[0]
        b: Tensor = op.inputs[1]
        c_shape = op.outputs[0].shape

        if len(a.shape) == 1:                    # shape: [a]
            a = a.unsqueeze([0, 1])              # [1, 1, a]
            if len(b.shape) == 2:                # shape: [a, b]
                # [a] x [a, b] -> [b]
                b = b.unsqueeze([0])             # [1, a, b]
                c = self.run_batch_matmul(a, b)  # [1, 1, b]
                c = c.squeeze([0, 1])            # [b]
            else:
                assert len(b.shape) >= 3                # shape example: [b, c, a, d]
                # [a] x [b, c, a, d] -> [b, c, d]
                b = b.flatten(start_dim=0, end_dim=-2)  # [b * c, a, d]
                c = self.run_batch_matmul(a, b)         # [b * c, 1, d]
                c = c.reshape(c_shape)                  # [b, c, d]
        elif len(b.shape) == 1:                  # shape: [b]
            b = b.unsqueeze([0, 2])              # [1, b, 1]
            if len(a.shape) == 2:                # shape: [a, b]
                a = a.unsqueeze([0])             # [1, a, b]
                c = self.run_batch_matmul(a, b)  # [1, a, 1]
                c = c.squeeze([0, 2])            # [a]
            else:
                assert len(a.shape) >= 3                # shape example: [a, c, d, b]
                # [a, c, d, b] x [b] -> [a, c, d]
                a = a.flatten(start_dim=0, end_dim=-2)  # [a * c, d, b]
                c = self.run_batch_matmul(a, b)         # [a * c, d, 1]
                c = c.reshape(c_shape)                  # [a, c, d]
        else:
            # example: [a, b, c] x [c, d] -> [a, b, d]
            assert len(a.shape) >= 2 and len(b.shape) >= 2
            if len(a.shape) == 2:
                a = a.unsqueeze([0])
            else:
                a = a.flatten(start_dim=0, end_dim=-2)
            if len(b.shape) == 2:
                b = b.unsqueeze([0])
            else:
                b = b.flatten(start_dim=0, end_dim=-2)
            c = self.run_batch_matmul(a, b)
            c = c.reshape(c_shape)
        return [c]
