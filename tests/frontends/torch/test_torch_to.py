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
import torch
from torch._dynamo.exc import BackendCompilerFailed
from torch import nn


class TestTensorCpu(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.randn(1, device='cuda')

    def forward(self, x):
        return self.w.cpu() * x.cpu()


class TestTensorCuda(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.randn(1, device='cpu')

    def forward(self, x):
        return self.w.cuda() * x.cuda()


class TestTensorTo(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.randn(1, device='cpu')

    def forward(self, x):
        return self.w.to(device='cuda') * x.cuda()


def test_tensor_cpu():
    model = TestTensorCpu()
    model_opt = torch.compile(model, backend='hidet')

    x_cpu = torch.randn(10, device='cpu')
    model_opt(x_cpu)

    with pytest.raises(BackendCompilerFailed):
        x_cuda = torch.randn(10, device='cuda')
        model_opt(x_cuda)


def test_tensor_cuda():
    model = TestTensorCuda()
    model_opt = torch.compile(model, backend='hidet')

    x_cuda = torch.randn(10, device='cuda')
    model_opt(x_cuda)

    with pytest.raises(BackendCompilerFailed):
        x_cpu = torch.randn(10, device='cpu')
        model_opt(x_cpu)


def test_tensor_to():
    model = TestTensorTo()
    model_opt = torch.compile(model, backend='hidet')

    x_cuda = torch.randn(10, device='cuda')
    model_opt(x_cuda)

    with pytest.raises(BackendCompilerFailed):
        x_cpu = torch.randn(10, device='cpu')
        model_opt(x_cpu)
