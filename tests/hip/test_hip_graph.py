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
from typing import Tuple
import pytest
import numpy.testing
import hidet
import hidet.testing
from hidet.graph import FlowGraph, Tensor
from hidet.hip.graph import HipGraph


def example_graph() -> Tuple[FlowGraph, Tensor]:
    x = hidet.symbol([3, 4], device='hip')
    y = x + 3.0
    y = hidet.ops.square(x) - y
    y = hidet.ops.square(y) - x
    graph = hidet.trace_from(y, x)
    return graph, hidet.randn_like(x)


def test_flow_graph_hip_graph():
    if not hidet.hip.available():
        pytest.skip('HIP is not available')
    graph, x = example_graph()
    hip_graph: HipGraph = graph.hip_graph()
    (actual,) = hip_graph.run(inputs=[x])
    expected = graph(x)
    numpy.testing.assert_allclose(actual=actual.cpu().numpy(), desired=expected.cpu().numpy(), atol=0.0, rtol=0.0)


def test_compiled_graph_hip_graph():
    if not hidet.hip.available():
        pytest.skip('HIP is not available')
    graph, x = example_graph()
    hip_graph: HipGraph = graph.build().hip_graph()
    (actual,) = hip_graph.run(inputs=[x])
    expected = graph(x)
    numpy.testing.assert_allclose(actual=actual.cpu().numpy(), desired=expected.cpu().numpy(), atol=0.0, rtol=0.0)


if __name__ == '__main__':
    pytest.main([__file__])
