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
from typing import List, Optional, Callable, Any
from functools import lru_cache

import hidet.cuda
from hidet.ir import dtypes
from hidet.graph.ir import Operator, Tensor
from hidet.graph.transforms import ResolveRule, register_resolve_rule
from hidet.utils.py import gcd, factorize, prod, cdiv

from .matmul import MatmulOp
from .batch_matmul import batch_matmul
from .matmul_f16 import matmul_f16
from ..transform import broadcast, flatten
from ..utils import broadcast_shapes


def parallel_k_heuristic_nparts(batch_size, m_size, n_size, k_size) -> int:
    estimate_thread_blocks = batch_size * ((m_size + 63) // 64) * ((n_size + 63) // 64)
    num_multi_processors = hidet.cuda.properties().multiProcessorCount
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
    nparts_candidates = [nparts for nparts in factorize(k_size) if nparts <= 16]
    best_nparts = None
    best_nparts_latency = 1e9
    latencies = []

    print(
        'search parallel k factor for [{} x {} x {} x {}] among {}'.format(
            batch_size, m_size, n_size, k_size, nparts_candidates
        )
    )
    for nparts in nparts_candidates:
        a = hidet.symbol([batch_size, m_size, k_size], dtype=dtype, device='cuda')
        b = hidet.symbol([batch_size, k_size, n_size], dtype=dtype, device='cuda')
        # to [batch_size * nparts, m_size, k_size // nparts]
        aa = a.reshape([batch_size, m_size, nparts, k_size // nparts]).rearrange([[0, 2], [1], [3]])
        # to [batch_size * nparts, k_size // nparts, n_size]
        bb = b.reshape([batch_size, nparts, k_size // nparts, n_size]).rearrange([[0, 1], [2], [3]])
        cc = batch_matmul(aa, bb, mma=mma)
        c = cc.reshape([batch_size, nparts, m_size, n_size]).sum(1)

        graph: hidet.FlowGraph = hidet.trace_from(c, [a, b])
        # with hidet.graph.PassContext() as ctx:
        # graph: hidet.FlowGraph = hidet.graph.transforms.fuse_operator_pass()(graph)
        graph: hidet.FlowGraph = hidet.graph.optimize(graph)

        latency: float = graph.latency()
        latencies.append(latency)
        if latency < best_nparts_latency:
            best_nparts = nparts
            best_nparts_latency = latency
    print(
        'got parallel k factor {} with latency {:.3f} ms ({})'.format(
            best_nparts, best_nparts_latency, ', '.join(['{:.3f}'.format(v) for v in latencies])
        )
    )
    return best_nparts


@register_resolve_rule(MatmulOp)
class MatmulResolveRule(ResolveRule):
    """
    Resolve a generic matrix multiplication operator to a batched matrix multiplication operator.

    The generic matrix multiplication operator has the same semantics as numpy.matmul that accepts
    variable dimensions of inputs.

    On ther other hand, the batched matrix multiplication operator accepts inputs with shape:
    [batch_size, m_size, k_size] x [batch_size, k_size, n_size]

    This resolve rule also parallelize k dimension when possible, and determine the mma instruction.
    """

    def run_batch_matmul(self, a: Tensor, b: Tensor) -> Tensor:
        parallel_k = self.get_config('parallel_k', default='default')  # 'default', 'search', 2, 4, ...
        mma = self.get_config('mma', default='simt')  # 'simt', 'mma', 'wmma'

        batch_size, m_size, n_size, k_size = a.shape[0], a.shape[1], b.shape[2], a.shape[2]
        if parallel_k == 'default':
            nparts = parallel_k_heuristic_nparts(batch_size, m_size, n_size, k_size)
        elif parallel_k == 'search':
            nparts = parallel_k_search_nparts(a.dtype.name, mma, batch_size, m_size, n_size, k_size)
        elif parallel_k == 'disabled':
            nparts = 1
        elif isinstance(parallel_k, int):
            nparts = gcd(parallel_k, k_size)
        else:
            raise ValueError(f'invalid parallel_k: {parallel_k}')

        if nparts == 1:
            c = batch_matmul(a, b, mma=mma)
        else:
            # [batch_size * nparts, m_size, k_size // nparts]
            aa = a.reshape([batch_size, m_size, nparts, k_size // nparts]).rearrange([[0, 2], [1], [3]])
            # [batch_size * nparts, k_size // nparts, n_size]
            bb = b.reshape([batch_size, nparts, k_size // nparts, n_size]).rearrange([[0, 1], [2], [3]])
            c = batch_matmul(aa, bb, mma=mma).reshape([batch_size, nparts, m_size, n_size]).sum(1)
        return c

    def resolve_generic(self, op: Operator) -> Optional[List[Tensor]]:
        assert isinstance(op, MatmulOp)
        a: Tensor = op.inputs[0]
        b: Tensor = op.inputs[1]
        c_shape = list(op.outputs[0].shape)

        if len(a.shape) == 1:  # shape: [a]
            a = a.unsqueeze([0, 1])  # [1, 1, a]
            if len(b.shape) == 2:  # shape: [a, b]
                # [a] x [a, b] -> [b]
                b = b.unsqueeze([0])  # [1, a, b]
                c = self.run_batch_matmul(a, b)  # [1, 1, b]
                c = c.squeeze([0, 1])  # [b]
            else:
                assert len(b.shape) >= 3  # shape example: [b, c, a, d]
                # [a] x [b, c, a, d] -> [b, c, d]
                b = flatten(b, start_dim=0, end_dim=-3)  # [b * c, a, d]
                c = self.run_batch_matmul(a, b)  # [b * c, 1, d]
                c = c.reshape(c_shape)  # [b, c, d]
        elif len(b.shape) == 1:  # shape: [b]
            b = b.unsqueeze([0, 2])  # [1, b, 1]
            if len(a.shape) == 2:  # shape: [a, b]
                a = a.unsqueeze([0])  # [1, a, b]
                c = self.run_batch_matmul(a, b)  # [1, a, 1]
                c = c.squeeze([0, 2])  # [a]
            else:
                assert len(a.shape) >= 3  # shape example: [a, c, d, b]
                # [a, c, d, b] x [b] -> [a, c, d]
                a = flatten(a, start_dim=0, end_dim=-3)  # [a * c, d, b]
                c = self.run_batch_matmul(a, b)  # [a * c, d, 1]
                c = c.reshape(c_shape)  # [a, c, d]
        else:
            # example: [a, b, c] x [c, d] -> [a, b, d]
            assert len(a.shape) >= 2 and len(b.shape) >= 2
            a_head = list(a.shape[:-2])
            b_head = list(b.shape[:-2])
            c_head = broadcast_shapes([a_head, b_head, [1]])  # [1] is used to make sure len(c_head) > 0
            a_broadcast_shape = c_head + list(a.shape[-2:])
            b_broadcast_shape = c_head + list(b.shape[-2:])
            a = flatten(broadcast(a, a_broadcast_shape), start_dim=0, end_dim=-3)
            b = flatten(broadcast(b, b_broadcast_shape), start_dim=0, end_dim=-3)
            c = self.run_batch_matmul(a, b)
            c = c.reshape(c_shape)
        return [c]

    def resolve_f16(self, op: Operator) -> Optional[List[Tensor]]:
        a: Tensor = op.inputs[0]
        b: Tensor = op.inputs[1]
        c: Tensor = op.outputs[0]

        if not (a.dtype == dtypes.float16 and b.dtype == dtypes.float16 and a.shape[-1] % 8 == b.shape[-1] % 8 == 0):
            return None

        parallel_k = self.get_config('parallel_k', default='default')  # 'default', 'search', 2, 4, ...
        if isinstance(parallel_k, str):
            if parallel_k == 'default':
                batch_size, m_size, n_size, k_size = prod(c.shape[:-2]), c.shape[-2], c.shape[-1], a.shape[-1]
                estimate_blocks = batch_size * cdiv(m_size, 64) * cdiv(n_size, 64)
                estimate_concurrent_blocks = 80 * 5
                max_k_parts = cdiv(k_size, 64)
                k_parts = min(cdiv(estimate_concurrent_blocks, estimate_blocks), max_k_parts)
            elif parallel_k == 'disabled':
                k_parts = 1
            elif parallel_k == 'search':
                candidates = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16]
                aa = hidet.symbol_like(a)
                bb = hidet.symbol_like(b)
                latencies: List[float] = []
                print('Searching the best parallel_k for {} x {} among {}'.format(a.shape, b.shape, candidates))
                for candidate in candidates:
                    cc = matmul_f16(aa, bb, parallel_k_parts=candidate)
                    # if candidate > 1:
                    #     cc = cc.sum(0)
                    graph = hidet.trace_from([cc], [aa, bb])
                    graph: hidet.FlowGraph = hidet.graph.optimize(graph)
                    latency: float = graph.latency()
                    latencies.append(latency)
                best_idx = min(range(len(candidates)), key=lambda i: latencies[i])
                print(
                    'Results: {{{}}},'.format(
                        ', '.join('{}: {:.1f}'.format(a, b * 1000) for a, b in zip(candidates, latencies))
                    ),
                    'Picked {} with {:.1f} micro-seconds'.format(candidates[best_idx], latencies[best_idx] * 1000),
                )
                k_parts = candidates[best_idx]
            else:
                raise ValueError(f'invalid parallel_k: {parallel_k}')
        elif isinstance(parallel_k, int):
            k_parts = min(max(parallel_k, 1), 32)
        else:
            raise ValueError(f'invalid parallel_k: {parallel_k}')
        c = matmul_f16(a, b, parallel_k_parts=k_parts).sum(0)
        return [c]

    def resolve(self, op: Operator) -> Optional[List[Tensor]]:
        resolve_funcs: List[Callable[[Operator], Any]] = [self.resolve_f16, self.resolve_generic]
        for resolve_func in resolve_funcs:
            outs = resolve_func(op)
            if outs is not None:
                return outs
        return None
