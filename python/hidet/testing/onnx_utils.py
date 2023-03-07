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
from typing import Sequence

import numpy.testing
import torch

import hidet


def check_onnx_and_hidet(
    torch_model: torch.nn.Module, inputs: Sequence[torch.Tensor], atol=1e-4, rtol=1e-4, device='all'
):
    if device == 'all':
        devices = ['cuda', 'cpu']
        for dev in devices:
            check_onnx_and_hidet(torch_model, inputs, atol, rtol, dev)
        return

    # run torch
    device = torch.device(device)
    torch_model = torch_model.to(device).eval()
    inputs = [x.to(device) for x in inputs]
    torch_outputs = torch_model(*inputs)
    if isinstance(torch_outputs, torch.Tensor):
        torch_outputs = (torch_outputs,)

    # export to onnx
    onnx_path = hidet.utils.hidet_cache_file('./test_model.onnx')
    torch.onnx.export(torch_model, args=tuple(inputs), f=onnx_path)

    # run onnx via hidet
    onnx_model = hidet.frontend.from_onnx(onnx_path)
    hidet_inputs = [hidet.from_torch(x) for x in inputs]
    symbol_inputs = [hidet.symbol_like(x) for x in hidet_inputs]
    symbol_outputs = onnx_model(*symbol_inputs)
    graph = hidet.trace_from(symbol_outputs, inputs=symbol_inputs)

    # check outputs
    hidet_outputs = graph(*hidet_inputs)
    if isinstance(hidet_outputs, hidet.Tensor):
        hidet_outputs = (hidet_outputs,)
    for torch_output, hidet_output in zip(torch_outputs, hidet_outputs):
        torch_output = torch_output.detach().cpu().numpy()
        hidet_output = hidet_output.cpu().numpy()
        numpy.testing.assert_allclose(torch_output, hidet_output, atol=atol, rtol=rtol)
