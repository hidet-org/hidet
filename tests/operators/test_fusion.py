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
