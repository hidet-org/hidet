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


def check_module(model: torch.nn.Module, args: Sequence[torch.Tensor], atol=1e-4, rtol=1e-4):
    # print(type(model))
    with torch.no_grad():
        hidet.torch.dynamo_config.print_input_graph(True)
        model = torch.nn.Sequential(model)
        model = model.cuda()
        model.eval()
        args = [x.cuda() for x in args]
        model_opt = torch.compile(model, backend='hidet')
        torch_outputs = model(*args)
        hidet_outputs = model_opt(*args)
        if isinstance(torch_outputs, torch.Tensor):
            torch_outputs = (torch_outputs,)
        if isinstance(hidet_outputs, torch.Tensor):
            hidet_outputs = (hidet_outputs,)

        if len(torch_outputs) != len(hidet_outputs):
            raise ValueError('torch_outputs and hidet_outputs have different length')

        for torch_output, hidet_output in zip(torch_outputs, hidet_outputs):
            torch_output = torch_output.detach().cpu().numpy()
            hidet_output = hidet_output.detach().cpu().numpy()
            numpy.testing.assert_allclose(torch_output, hidet_output, atol=atol, rtol=rtol)
