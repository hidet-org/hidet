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
from hidet.graph.ops.fusion.fused_operator import fused_operator


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


def test_fusion_cublas_matmul():
    bs, m, n, k = 2, 1024, 1024, 1024
    a = hidet.symbol(shape=[bs, m, k], dtype='float32', device='cuda')
    b = hidet.randn(shape=[bs, k, n], dtype='float32', device='cuda')

    def optimize_and_build(op):
        c = op(a + 1.0, b) + 1.0
        graph = hidet.trace_from(c)
        graph_opt = hidet.graph.optimize(graph)
        compiled = graph_opt.build()
        return compiled

    graph_2 = optimize_and_build(hidet.ops.matmul_cublas)
    graph_1 = optimize_and_build(hidet.ops.batch_matmul)

    a = hidet.randn_like(a)

    y1 = graph_1(a)
    y2 = graph_2(a)

    hidet.utils.assert_close(y1, y2, atol=1e-5, rtol=1e-5)


if __name__ == '__main__':
    pytest.main([__file__])
