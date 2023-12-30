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

import numpy as np
from numpy import testing
import torch


def test_torch_to():
    DEVICE = 'cuda'

    class Model0(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, *args):
            to = args[0].to(dtype=torch.float32)
            return to

    model_0 = Model0()
    output_names_0 = ['v0_0']

    class Model1(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, *args):
            to = args[0].to(dtype=torch.float32)
            to_1 = args[0].to(dtype=torch.float32)
            return (to, to_1)

    model_1 = Model1()
    output_names_1 = ['v5_0', 'v0_0']

    data = np.random.rand(41).astype(np.float16)
    input_data_0 = [data]

    optmodel_0 = torch.compile(model_0, fullgraph=True, backend='hidet', mode=None)
    model_out_0 = optmodel_0(*[torch.from_numpy(v).to(DEVICE) for v in input_data_0])
    model_out_0 = (
        [v.to(DEVICE).detach() for v in model_out_0]
        if isinstance(model_out_0, tuple)
        else [model_out_0.to(DEVICE).detach()]
    )
    model_out_0 = [v.cpu().resolve_conj().numpy() if v.is_conj() else v.cpu().numpy() for v in model_out_0]
    output_0 = dict(zip(output_names_0, model_out_0))

    input_data_1 = [data]

    optmodel_1 = torch.compile(model_1, fullgraph=True, backend='hidet', mode=None)
    model_out_1 = optmodel_1(*[torch.from_numpy(v).to(DEVICE) for v in input_data_1])
    model_out_1 = (
        [v.to(DEVICE).detach() for v in model_out_1]
        if isinstance(model_out_1, tuple)
        else [model_out_1.to(DEVICE).detach()]
    )
    model_out_1 = [v.cpu().resolve_conj().numpy() if v.is_conj() else v.cpu().numpy() for v in model_out_1]
    output_1 = dict(zip(output_names_1, model_out_1))
    output_name_dict = {'v0_0': 'v0_0'}

    for tensor_name_0, tensor_name_1 in output_name_dict.items():
        testing.assert_allclose(
            output_0[tensor_name_0], output_1[tensor_name_1], rtol=1, err_msg=f'at {tensor_name_0}, {tensor_name_1}'
        )


if __name__ == '__main__':
    pytest.main([__file__])
