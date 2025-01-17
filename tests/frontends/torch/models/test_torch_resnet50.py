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
import torch.backends.cudnn
import pytest
from hidet.testing.torch_utils import check_module


@pytest.mark.parametrize('shape', [[1, 3, 224, 224]])
@pytest.mark.parametrize('dynamic', [False, True])
@pytest.mark.parametrize('dtype, tol', [(torch.float16, 2e-2), (torch.float32, 2e-2)])
def test_resnet18(shape, dynamic, dtype, tol, device):
    model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True).eval().to(dtype)
    x = torch.randn(*shape).cuda().to(dtype) * 0.1796 + 0.5491
    check_module(model, [x], atol=tol, rtol=tol, dynamic=dynamic, device=device)


@pytest.mark.slow
@pytest.mark.parametrize('shape', [[1, 3, 224, 224]])
@pytest.mark.parametrize('dynamic', [False, True])
@pytest.mark.parametrize('dtype, tol', [(torch.float16, 2e-2), (torch.float32, 1e-4)])
def test_resnet50(shape, dynamic, dtype, tol, device):
    model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet50', pretrained=True).eval().to(dtype)
    x = torch.randn(*shape).cuda().to(dtype) * 0.1796 + 0.5491
    check_module(model, [x], atol=tol, rtol=tol, dynamic=dynamic, device=device)


if __name__ == '__main__':
    pytest.main([__file__])
