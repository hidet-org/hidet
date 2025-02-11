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
from hidet.ir.expr import is_constant
from hidet.graph.tensor import Tensor
from hidet.graph.operator import Operator
from hidet.graph.transforms import ResolveRule, register_resolve_rule
from hidet.utils.py import gcd, factorize

from .matmul import MatmulOp
from .batch_matmul import batch_matmul
from .matmul_f16_cute import matmul_f16_cute as matmul_f16_cute_stable
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
        parallel_k = hidet.option.get_parallel_k()
        mma = self.get_config('mma', default='mma')  # 'simt', 'mma'

        if any(not isinstance(v, int) for v in a.shape + b.shape):
            nparts = 1
        else:
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
        if a.dtype.nbytes > 4 or b.dtype.nbytes > 4:
            return None

        # If either a or b is a tensor with actual storage(i.e., not symbolic),
        # the operator `flatten` calls in the code below will
        # require additional memory and somehow these allocated spaces are not released during the model compilation.
        # This causes the error described in issue #326.

        exection_mode = hidet.option.get_execution_mode()
        if a.is_symbolic() or b.is_symbolic():
            hidet.option.execution_mode('symbolic')

        if op.attrs['transpose_b']:
            b = b.transpose(-2, -1)

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

        hidet.option.execution_mode(exection_mode)
        return [c]

    def resolve_f16(self, op: Operator) -> Optional[List[Tensor]]:
        if op.attrs['require_prologue']:
            return None
        # if op.task.has_symbolic_shape():
        #     return None

        a: Tensor = op.inputs[0]
        b: Tensor = op.inputs[1]
        c: Tensor = op.outputs[0]

        transpose_b = op.attrs['transpose_b']

        if not transpose_b and not (
            a.dtype.is_any_float16()
            and b.dtype.is_any_float16()
            and is_constant(a.shape[-1], b.shape[-1])
            and (a.shape[-1] % 2 == b.shape[-1] % 2 == 0)
        ):
            return None

        elif transpose_b and not (
            a.dtype.is_any_float16()
            and b.dtype.is_any_float16()
            and is_constant(a.shape[-1], b.shape[-2])
            and (a.shape[-1] % 2 == b.shape[-2] % 2 == 0)
        ):
            return None

        if hidet.option.cuda.get_arch_pair() < (8, 0):
            return None

        hexcute_matmul = hidet.option.get_hexcute_matmul()
        if hexcute_matmul == 'enable':
            from .matmul_f16_cute_experimental import matmul_f16_cute as matmul_f16_cute_experimental

            matmul_f16_cute = matmul_f16_cute_experimental
        elif hexcute_matmul == 'disable':
            matmul_f16_cute = matmul_f16_cute_stable
        else:
            # Leave this to be implemented in the future
            raise NotImplementedError('The heuristic for hexcute_matmul is not implemented.')

        c = matmul_f16_cute(a, b, transpose_b=transpose_b)
        return [c]

    def resolve(self, op: Operator) -> Optional[List[Tensor]]:
        if op.device.is_cpu():
            return None
        # TODO: support amd gpu
        if not hidet.cuda.available():
            return None
        resolve_funcs: List[Callable[[Operator], Any]] = [self.resolve_f16, self.resolve_generic]
        for resolve_func in resolve_funcs:
            outs = resolve_func(op)
            if outs is not None:
                return outs
        return None
