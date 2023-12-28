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
import pytest
from hidet.testing.torch_utils import check_module


@pytest.mark.parametrize('shape', [[1, 3, 224, 224]])
def test_densenet121(shape):
    model = torch.hub.load('pytorch/vision:v0.6.0', 'densenet121', pretrained=True).cuda().eval().to(torch.float16)
    x = torch.randn(*shape).cuda().to(torch.float16) * 0.1796 + 0.5491
    check_module(model, [x], atol=2e-2, rtol=2e-2, dynamic=False)


if __name__ == '__main__':
    pytest.main([__file__])
