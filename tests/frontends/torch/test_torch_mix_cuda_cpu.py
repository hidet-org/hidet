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
import hidet
import hidet.testing
import torch
from torch import nn


class CopyTensorModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.w_cpu = torch.zeros(1, device='cpu')
        self.w_cuda = torch.zeros(1, device='cuda')

    def forward(self, x: torch.Tensor):
        return ((x.cpu() + self.w_cpu).cuda() + self.w_cuda).cpu().cuda()


def test_torch_mix_cuda_cpu(device):
    if device != 'cuda':
        pytest.skip('TODO: support hip backend')
    model = CopyTensorModule()
    x = torch.randn(3, 4, device='cpu')
    y = model(x)

    model_opt = torch.compile(model, backend='hidet', mode=None)
    y1 = model_opt(x)

    torch.testing.assert_close(y, y1, rtol=0.0, atol=0.0)
