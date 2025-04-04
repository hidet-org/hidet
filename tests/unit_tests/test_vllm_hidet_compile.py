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
import hidet
from hidet.graph.frontend.torch.dynamo_backends import *
from hidet.graph.frontend.torch.dynamo_config import dynamo_config

import torch
from torch import nn
from torch._subclasses.fake_tensor import FakeTensorMode

from hidet.testing import device_to_torch


class TwoMatmul(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(4096, 11008, bias=False)
        self.lin2 = nn.Linear(4096, 11008, bias=False)

    def forward(self, x):
        y1 = self.lin1(x)
        y2 = self.lin2(x)
        return y1, y2


def test_compile_with_fake_tensor(device):
    dynamo_config.use_cuda_graph(False)
    torch_device = device_to_torch(device)

    graph_modules = []

    def get_graph_module(graph_module, example_inputs, **kwargs):
        nonlocal graph_modules
        graph_modules.append(graph_module)
        return graph_module

    model = TwoMatmul().to(torch.half).cuda().eval()
    x = torch.randn(1, 47, 4096, device=torch_device, dtype=torch.half)

    with torch.inference_mode(True):
        compiled_model = torch.compile(model, backend=get_graph_module, mode='max-autotune')
        compiled_model(x)

    fake_inputs = []
    real_inputs = []
    interpreter = hidet.frontend.from_torch(graph_modules[-1])
    for fxgraph_node in interpreter.graph.nodes:
        if fxgraph_node.op != 'placeholder':
            continue
        example_value = fxgraph_node.meta['example_value']
        fake_inputs.append(example_value)
        real_inputs.append(torch.randn(example_value.shape, dtype=example_value.dtype, device=example_value.device))

    flow_graph, inputs, traceable_input_ids, output_format = get_flow_graph(interpreter, fake_inputs)
    cgraph = get_compiled_graph(flow_graph, {})
    hidet_fake_compiled = HidetCompiledModel(cgraph, inputs, traceable_input_ids, output_format)

    flow_graph, inputs, traceable_input_ids, output_format = get_flow_graph(interpreter, real_inputs)
    cgraph = get_compiled_graph(flow_graph, {})
    hidet_real_compiled = HidetCompiledModel(cgraph, inputs, traceable_input_ids, output_format)

    f = hidet_fake_compiled(*real_inputs)
    r = hidet_real_compiled(*real_inputs)
    torch.testing.assert_close(r, f, atol=0.0001, rtol=0.0001)
