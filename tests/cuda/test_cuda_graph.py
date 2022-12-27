from typing import Tuple
import pytest
import numpy.testing
import hidet
import hidet.testing
from hidet.graph import FlowGraph, Tensor
from hidet.cuda.graph import CudaGraph


def example_graph() -> Tuple[FlowGraph, Tensor]:
    x = hidet.symbol([3, 4])
    y = x + 3.0
    y = hidet.ops.square(x) - y
    y = hidet.ops.square(y) - x
    graph = hidet.trace_from(y, x)
    return graph, hidet.randn_like(x)


def test_cuda_graph():
    graph, x = example_graph()
    cuda_graph: CudaGraph = graph.cuda_graph()
    (actual,) = cuda_graph.run(inputs=[x])
    expected = graph(x)
    numpy.testing.assert_allclose(actual=actual.cpu().numpy(), desired=expected.cpu().numpy(), atol=0.0, rtol=0.0)


if __name__ == '__main__':
    pytest.main([__file__])
