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

from hidet.testing.torch_utils import FunctionalModule, check_module


@pytest.mark.parametrize('a_shape', [[1, 3, 64], [10, 10], [11, 13], [1, 2, 3]])
@pytest.mark.parametrize('sizes', [[1, 2, 3], [2, 3, 4, 5, 6, 8]])
def test_tensor_repeat(a_shape, sizes, device):
    def tensor_repeat(tensor):
        return tensor.repeat(*sizes)

    check_module(FunctionalModule(op=tensor_repeat), args=[torch.randn(a_shape)], atol=0, rtol=0, device=device)


@pytest.mark.parametrize('a, b', [[[1, 3, 2], [1, 3, 2]], [2.0, [10, 10]], [[11, 13], 2]])
def test_pow(a, b, device):
    if isinstance(a, list) and not isinstance(b, list):
        args = [torch.randn(a), b]
    elif isinstance(b, list) and not isinstance(a, list):
        args = [a, torch.randn(b)]
    else:
        args = [torch.randn(a), torch.randn(b)]

    check_module(FunctionalModule(op=torch.pow), args=args, atol=0.0001, rtol=0.0001, device=device)


if __name__ == '__main__':
    pytest.main([__file__])
