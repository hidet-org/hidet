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
import torch
import numpy
import pytest
import hidet
from hidet.utils.counters import counters
from hidet.graph.frontend.torch.flow_graph_cache import (
    compute_flow_graph_hash,
    flow_graph_cache_clear,
    compiled_graph_in_memory_cache,
)


@pytest.fixture(autouse=True)
def setup():
    # Skip non-CUDA tests
    if not torch.cuda.is_available():
        pytest.skip("CUDA device not available")
    # clear in meory cache
    compiled_graph_in_memory_cache.clear()
    # clear disk cache
    flow_graph_cache_clear()
    counters.clear()
    # Reset state before each test
    torch._dynamo.reset()
    yield


@pytest.mark.parametrize("device", ["cuda"])
@pytest.mark.parametrize("size", [(8, 8)])
@pytest.mark.parametrize("dtype", [torch.float16])
def test_flowgraph_cache_hit(size, device, dtype):
    # test cache hit cases
    a = torch.rand(size=size, dtype=dtype).to(device)
    b = torch.rand(size=size, dtype=dtype).to(device)

    def fn(x):
        return x @ x

    compiled_fn = torch.compile(fn, backend='hidet', mode="max-autotune-no-cudagraphs")
    numpy.testing.assert_allclose(fn(a).detach().cpu().numpy(), compiled_fn(a).detach().cpu().numpy(), atol=1e-2)
    assert counters['flow_graph_cache']['miss'] == 1
    assert counters['flow_graph_cache']['hit'] == 0
    # reset dynamo to avoid guarding
    torch._dynamo.reset()
    counters.clear()
    # same input should hit cache
    numpy.testing.assert_allclose(fn(a).detach().cpu().numpy(), compiled_fn(a).detach().cpu().numpy(), atol=1e-2)
    assert counters['flow_graph_cache']['miss'] == 0
    assert counters['flow_graph_cache']['hit'] == 1
    torch._dynamo.reset()
    counters.clear()
    # same input shape and dtype but different tensor values should hit cache
    numpy.testing.assert_allclose(fn(b).detach().cpu().numpy(), compiled_fn(b).detach().cpu().numpy(), atol=1e-2)
    assert counters['flow_graph_cache']['miss'] == 0
    assert counters['flow_graph_cache']['hit'] == 1


@pytest.mark.parametrize("device", ["cuda"])
@pytest.mark.parametrize("size", [(8, 8)])
@pytest.mark.parametrize("dtype", [torch.float16])
def test_flowgraph_cache_miss(size, device, dtype):
    # test cache hits cases
    a = torch.rand(size=size, dtype=dtype).to(device)
    b = torch.rand(size=[4, 4], dtype=dtype).to(device)
    c = torch.rand(size=size, dtype=torch.float32).to(device)

    def fn(x):
        return x @ x

    compiled_fn = torch.compile(fn, backend='hidet', mode="max-autotune-no-cudagraphs")
    numpy.testing.assert_allclose(fn(a).detach().cpu().numpy(), compiled_fn(a).detach().cpu().numpy(), atol=1e-2)
    assert counters['flow_graph_cache']['miss'] == 1
    assert counters['flow_graph_cache']['hit'] == 0
    torch._dynamo.reset()
    counters.clear()
    # different input shape should miss cache and trigger recompilation
    numpy.testing.assert_allclose(fn(b).detach().cpu().numpy(), compiled_fn(b).detach().cpu().numpy(), atol=1e-2)
    assert counters['flow_graph_cache']['miss'] == 1
    assert counters['flow_graph_cache']['hit'] == 0
    torch._dynamo.reset()
    counters.clear()
    # different input dtype should miss cache and trigger recompilation
    numpy.testing.assert_allclose(fn(c).detach().cpu().numpy(), compiled_fn(c).detach().cpu().numpy(), atol=1e-2)
    assert counters['flow_graph_cache']['miss'] == 1
    assert counters['flow_graph_cache']['hit'] == 0
    torch._dynamo.reset()
    counters.clear()
    compiled_fn_1 = torch.compile(fn, backend='hidet', mode="max-autotune")
    # different configuration should miss cache and trigger recompilation
    numpy.testing.assert_allclose(fn(a).detach().cpu().numpy(), compiled_fn_1(a).detach().cpu().numpy(), atol=1e-2)
    assert counters['flow_graph_cache']['miss'] == 1
    assert counters['flow_graph_cache']['hit'] == 0


@pytest.mark.parametrize("dtype", [torch.float16])
def test_flowgraph_dynamic(dtype):
    a = torch.rand(size=[8, 8], dtype=dtype).to('cuda')
    b = torch.rand(size=[16, 8], dtype=dtype).to('cuda')
    c = torch.rand(size=[8, 16], dtype=dtype).to('cuda')

    def fn(x):
        return x * 2

    torch._dynamo.mark_dynamic(a, 0)
    compiled_fn = torch.compile(fn, backend='hidet', mode="max-autotune-no-cudagraphs")
    numpy.testing.assert_allclose(fn(a).detach().cpu().numpy(), compiled_fn(a).detach().cpu().numpy(), atol=1e-2)
    assert counters['flow_graph_cache']['miss'] == 1
    assert counters['flow_graph_cache']['hit'] == 0
    torch._dynamo.reset()
    counters.clear()
    torch._dynamo.mark_dynamic(b, 0)
    # For dynamic input, batch size dimension variation should hit cache
    numpy.testing.assert_allclose(fn(b).detach().cpu().numpy(), compiled_fn(b).detach().cpu().numpy(), atol=1e-2)
    assert counters['flow_graph_cache']['miss'] == 0
    assert counters['flow_graph_cache']['hit'] == 1
    torch._dynamo.reset()
    counters.clear()
    torch._dynamo.mark_dynamic(c, 1)
    # different dynamic dimension should miss cache
    numpy.testing.assert_allclose(fn(c).detach().cpu().numpy(), compiled_fn(c).detach().cpu().numpy(), atol=1e-2)
    assert counters['flow_graph_cache']['miss'] == 1
    assert counters['flow_graph_cache']['hit'] == 0


def test_flowgraph_constant_input():
    # Same graph with different constant input should produce different graph hash
    x = hidet.symbol([10, 10])
    bias0 = hidet.randn([10, 10])
    bias1 = hidet.randn([10, 10])
    y = x + bias0
    graph_hash_one = compute_flow_graph_hash(hidet.trace_from(y, inputs=[x]), {})
    y = x + bias1
    graph_hash_two = compute_flow_graph_hash(hidet.trace_from(y, inputs=[x]), {})
    assert graph_hash_one != graph_hash_two
