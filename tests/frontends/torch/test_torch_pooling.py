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
from hidet.testing.torch_utils import check_module


@pytest.mark.parametrize('shape', [[1, 3, 224, 224]])
@pytest.mark.parametrize('kernel_size', [3, 3])
@pytest.mark.parametrize('stride', [2])
@pytest.mark.parametrize('padding', [1])
@pytest.mark.parametrize('dtype', [torch.float32])
def test_average_pool_2d(shape, kernel_size, stride, padding, dtype):
    check_module(
        torch.nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding), [torch.randn(shape, dtype=dtype)]
    )


@pytest.mark.parametrize('shape', [[1, 3, 224, 224]])
@pytest.mark.parametrize('kernel_size', [3, 3])
@pytest.mark.parametrize('stride', [2])
@pytest.mark.parametrize('padding', [1])
@pytest.mark.parametrize('dtype', [torch.float32])
def test_max_pool_2d(shape, kernel_size, stride, padding, dtype):
    check_module(
        torch.nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding), [torch.randn(shape, dtype=dtype)]
    )


@pytest.mark.parametrize('shape', [[1, 3, 8, 224, 224]])
@pytest.mark.parametrize('kernel_size', [2, 3, 3])
@pytest.mark.parametrize('stride', [2])
@pytest.mark.parametrize('padding', [1])
@pytest.mark.parametrize('dtype', [torch.float32])
def test_max_pool_3d(shape, kernel_size, stride, padding, dtype):
    check_module(
        torch.nn.MaxPool3d(kernel_size=kernel_size, stride=stride, padding=padding), [torch.randn(shape, dtype=dtype)]
    )


if __name__ == '__main__':
    pytest.main([__file__])
