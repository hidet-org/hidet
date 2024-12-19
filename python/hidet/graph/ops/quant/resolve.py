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
from typing import Optional, List
import hidet
from hidet.graph.operator import Operator, Tensor
from hidet.graph.transforms import ResolveRule, register_resolve_rule
from hidet.ir.expr import is_constant
from hidet.utils.py import cdiv, prod

from .matmul import SymmetricQuantizedMatmulOp
from .matmul_f16_i8 import symmetric_quant_matmul_f16_i8


@register_resolve_rule(SymmetricQuantizedMatmulOp)
class QuantSymmetricResolveRule(ResolveRule):
    def resolve(self, op: Operator) -> Optional[List[Tensor]]:
        # if op.task.has_symbolic_shape():
        #     return None

        a: Tensor = op.inputs[0]
        b: Tensor = op.inputs[1]
        scale: Tensor = op.inputs[2]
        c: Tensor = op.outputs[0]

        if hidet.option.cuda.get_arch_pair() < (8, 0):
            return None

        parallel_k = hidet.option.get_parallel_k()

        if not (is_constant(a.shape[-1]) and is_constant(b.shape[-2])):
            k_parts = 1
        elif isinstance(parallel_k, str):
            if parallel_k == 'default':
                batch_size, m_size, n_size, k_size = prod(c.shape[:-2]), c.shape[-2], c.shape[-1], a.shape[-1]
                if is_constant(batch_size, m_size):
                    estimate_blocks = batch_size * cdiv(m_size, 64) * cdiv(n_size, 64)
                    estimate_concurrent_blocks = 80 * 5
                    max_k_parts = cdiv(k_size, 64)
                    k_parts = min(cdiv(estimate_concurrent_blocks, estimate_blocks), max_k_parts)
                else:
                    k_parts = 1
            elif parallel_k == 'disabled':
                k_parts = 1
            elif parallel_k == 'search':
                candidates = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16]
                # to get around vcuda error
                # temporary hack to sample latency from symbolic shape
                a_shape = list(a.shape)
                b_shape = list(b.shape)
                for i in range(len(a_shape)):
                    if not is_constant(a_shape[i]):
                        a_shape[i] = 1
                for i in range(len(b_shape)):
                    if not is_constant(b_shape[i]):
                        b_shape[i] = 1

                aa = hidet.symbol(a_shape, dtype=a.dtype, device='cuda')
                bb = hidet.symbol(b_shape, dtype=b.dtype, device='cuda')
                sscale = hidet.symbol_like(scale, device='cuda')

                latencies: List[float] = []
                print(
                    'Symmetric Quantized Matmul: Searching the best parallel_k for {} x {} x {} among {}'.format(
                        a.shape, b.shape, scale.shape, candidates
                    )
                )
                for candidate in candidates:
                    cc = symmetric_quant_matmul_f16_i8(aa, bb, sscale, parallel_k_parts=candidate)
                    cc = cc.sum(0)
                    graph = hidet.trace_from([cc], [aa, bb, sscale])
                    # prevent recursion
                    with hidet.graph.PassContext():
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
        c = symmetric_quant_matmul_f16_i8(a, b, scale, parallel_k_parts=k_parts).sum(0)
        return [c]
