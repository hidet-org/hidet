import pytest
import hidet
import numpy


def model() -> hidet.FlowGraph:
    a = hidet.symbol([1, 3, 4], device='cuda')
    b = hidet.symbol([1, 4, 5], device='cuda')
    c = hidet.symbol([1, 3, 5], device='cuda')
    d = hidet.ops.batch_matmul(a, b) + c
    graph = hidet.trace_from(d, [a, b, c])
    return graph


def test_fusion():
    from hidet.graph.ops.definitions.fusion.fused_operator import fused_operator

    graph = model()
    a = hidet.randn([1, 3, 4], device='cuda')
    b = hidet.randn([1, 4, 5], device='cuda')
    c = hidet.randn([1, 3, 5], device='cuda')
    cc1 = fused_operator(a, b, c, fused_graph=graph, anchor=0)
    cc2 = graph(a, b, c)

    numpy.testing.assert_allclose(cc1.cpu().numpy(), cc2.cpu().numpy())


if __name__ == '__main__':
    pytest.main([__file__])
