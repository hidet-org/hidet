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
from typing import Optional, Union
import pytest
import numpy as np
import numpy.testing
import hidet
import hidet.testing
from hidet.graph import FlowGraph
from hidet import ops
import torch
from hidet.ffi import runtime_api
from hidet.graph.frontend.torch.utils import Placeholder, deserialize_output
from hidet.ir.expr import SymbolVar
from hidet.ir.type import data_type


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_attention(device):
    if device == 'cuda':
        pytest.skip(
            'This test has unstable numerical error.'
            'Issue https://github.com/CentML/hidet/issues/605 to investigate it.'
        )
    wte = hidet.randn([50257, 768], device=device)
    wpe = hidet.randn([1024, 768], device=device)
    w1 = hidet.randn([768, 768 * 3], device=device)
    b1 = hidet.randn([768 * 3], device=device)

    def get_graph(seq: Union[int, str]) -> FlowGraph:
        n_head = 12
        ids = hidet.symbol([seq], dtype='int32', device=device)
        x = hidet.ops.take(wte, ids) + hidet.ops.take(wpe, hidet.ops.arange(ids.shape[0], device=device))
        causal_mask = (1 - hidet.ops.tri(x.shape[0], dtype=x.dtype, device=x.device)) * -1e10  # [n_seq, n_seq]
        x = hidet.ops.matmul(x, w1) + b1
        x = ops.reshape(x, [x.shape[0], 3, n_head, x.shape[1] // (3 * n_head)])
        x = ops.transpose(x, [1, 2, 0, 3])
        q, k, v = [t for t in ops.split(x, 3, axis=0)]
        x = ops.softmax(q @ ops.transpose(k, [-1, -2]) / float(np.sqrt(q.shape[-1])) + causal_mask, axis=-1) @ v
        return hidet.trace_from(x)

    graph_dynamic = get_graph('seq')
    graph_dynamic_opt = hidet.graph.optimize(graph_dynamic)

    for seq in [1, 2, 3, 4, 8]:
        graph_static = get_graph(seq)

        x = hidet.randn([seq], dtype='int32', device=device)
        y_static = graph_static(x)
        y_dynamic = graph_dynamic(x)
        y_dynamic_opt = graph_dynamic_opt(x)
        for y in [y_dynamic, y_dynamic_opt]:
            numpy.testing.assert_allclose(y_static.cpu().numpy(), y.cpu().numpy(), atol=2e-1, rtol=2e-1)


@pytest.mark.parametrize('bs,h,w', [(1, 224, 224), (2, 224, 224), (1, 256, 256)])
def test_resnet50(device, bs, h, w):
    model = hidet.testing.models.resnet50().to(device=device)
    x = hidet.symbol(['bs', 3, 'h', 'w'], device=device)
    y = model(x)
    graph_static = hidet.trace_from(model(hidet.symbol([bs, 3, h, w], device=device)))
    graph_dynamic = hidet.trace_from(y)
    graph_dynamic_opt = hidet.graph.optimize(graph_dynamic)
    xx = hidet.randn([bs, 3, h, w], device=device, mean=0.45, stddev=0.22)
    y1 = graph_static(xx)
    y2 = graph_dynamic(xx)
    y3 = graph_dynamic_opt(xx)
    # we used random weights, thus the tolerance is larger than 1e-5
    numpy.testing.assert_allclose(y1.cpu().numpy(), y2.cpu().numpy(), rtol=5e-4, atol=5e-4)
    numpy.testing.assert_allclose(y1.cpu().numpy(), y3.cpu().numpy(), rtol=5e-2, atol=5e-2)


# @pytest.mark.parametrize('bs,h,w', [(1, 224, 224), (2, 224, 224), (1, 256, 256)])
def test_deserialization():
    runtime_api.set_symbol_value('s0', 4)

    s0 = SymbolVar('s0', dtype=data_type('int32'))
    tensors = [torch.ones(1, 1), torch.ones(2, 2)]

    format = [Placeholder(0), (s0, 20), s0, Placeholder(1)]
    outputs = deserialize_output(format, tensors)
    assert outputs[0] is tensors[0]
    assert outputs[1] == (4, 20)
    assert outputs[2] == 4
    assert outputs[3] is tensors[1]

    format = [Placeholder(1), (20, s0), Placeholder(0), s0]
    outputs = deserialize_output(format, tensors)
    assert outputs[0] is tensors[1]
    assert outputs[1] == (20, 4)
    assert outputs[2] is tensors[0]
    assert outputs[3] == 4
