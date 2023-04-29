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
import pytest
import numpy
import hidet
from hidet.graph.ops.definitions.fusion.fused_operator import fused_operator


def test_fusion():
    def model() -> hidet.FlowGraph:
        a = hidet.symbol([1, 3, 4], device='cuda')
        b = hidet.symbol([1, 4, 5], device='cuda')
        c = hidet.symbol([1, 3, 5], device='cuda')
        d = hidet.ops.batch_matmul(a, b) + c
        return hidet.trace_from(d, [a, b, c])

    graph = model()
    a = hidet.randn([1, 3, 4], device='cuda')
    b = hidet.randn([1, 4, 5], device='cuda')
    c = hidet.randn([1, 3, 5], device='cuda')
    cc1 = fused_operator(a, b, c, fused_graph=graph, anchor=0)
    cc2 = graph(a, b, c)

    numpy.testing.assert_allclose(cc1.cpu().numpy(), cc2.cpu().numpy())


def test_fusion_v2():
    def model() -> hidet.FlowGraph:
        a = hidet.symbol([1, 9], device='cuda')
        b = hidet.symbol([], device='cuda')
        c = hidet.ops.equal(a, b)
        d = hidet.ops.logical_not(c)
        e = d.astype('int32')
        f = hidet.ops.cumsum(e, dim=1)
        g = f * e
        h = g.astype('int64')
        i = h + 1
        return hidet.trace_from(i, [a, b])

    graph = model()
    a = hidet.zeros([1, 9], device='cuda')
    b = hidet.zeros([], device='cuda')
    y1 = graph(a, b)
    y2 = fused_operator(a, b, fused_graph=graph)

    numpy.testing.assert_allclose(y1.cpu().numpy(), y2.cpu().numpy())


if __name__ == '__main__':
    pytest.main([__file__])
