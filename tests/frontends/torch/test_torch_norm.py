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


@pytest.mark.parametrize('shape', [[2, 2], [2, 6]])
@pytest.mark.parametrize('num_features', [2, 2])
@pytest.mark.parametrize('dtype', [torch.float32])
def test_instance_norm_1d(shape, num_features, dtype):
    check_module(torch.nn.InstanceNorm1d(num_features=num_features), [torch.randn(shape, dtype=dtype)])


@pytest.mark.parametrize('shape', [[2, 2, 2], [2, 6, 4]])
@pytest.mark.parametrize('num_features', [2, 2])
@pytest.mark.parametrize('dtype', [torch.float32])
def test_instance_norm_2d(shape, num_features, dtype):
    check_module(torch.nn.InstanceNorm2d(num_features=num_features), [torch.randn(shape, dtype=dtype)])


@pytest.mark.parametrize('shape', [[2, 2, 2, 2], [2, 6, 4, 6]])
@pytest.mark.parametrize('num_features', [2, 2])
@pytest.mark.parametrize('dtype', [torch.float32])
def test_instance_norm_3d(shape, num_features, dtype):
    check_module(torch.nn.InstanceNorm3d(num_features=num_features), [torch.randn(shape, dtype=dtype)])


@pytest.mark.parametrize('shape', [[2, 2]])
@pytest.mark.parametrize('normalized_shape', [2])
@pytest.mark.parametrize('dtype', [torch.float32])
def test_layer_norm(shape, normalized_shape, dtype):
    check_module(torch.nn.LayerNorm(normalized_shape=normalized_shape), [torch.randn(shape, dtype=dtype)])


@pytest.mark.parametrize('shape', [[2, 2], [2, 2]])
@pytest.mark.parametrize('num_groups', [2, 2])
@pytest.mark.parametrize('num_channels', [2, 2])
@pytest.mark.parametrize('dtype', [torch.float32])
def test_group_norm(shape, num_groups, num_channels, dtype):
    check_module(
        torch.nn.GroupNorm(num_groups=num_groups, num_channels=num_channels), [torch.randn(shape, dtype=dtype)]
    )


if __name__ == '__main__':
    pytest.main([__file__])
