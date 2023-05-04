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
from typing import Optional, Tuple, List
import pytest
import torch
from hidet.testing.torch_utils import check_module


@pytest.mark.parametrize('shape', [[2, 2]])
@pytest.mark.parametrize('normalized_shape', [2])
@pytest.mark.parametrize('dtype', [torch.float32])
def test_layer_norm(shape, normalized_shape, dtype):
    check_module(torch.nn.LayerNorm(normalized_shape=normalized_shape), [torch.randn(shape, dtype=dtype)])


@pytest.mark.parametrize('shape', [[1, 4, 32, 32]])
@pytest.mark.parametrize('num_groups', [1, 2, 4])
@pytest.mark.parametrize('dtype', [torch.float32])
def test_group_norm(shape, num_groups, dtype):
    check_module(torch.nn.GroupNorm(num_groups=num_groups, num_channels=shape[1]), [torch.randn(shape, dtype=dtype)])


@pytest.mark.parametrize(
    "input_size, size, scale_factor, mode",
    [
        [[1, 3, 32, 32], (64, 64), None, 'nearest'],
        [[1, 3, 32, 32], None, 1.3, 'nearest'],
        [[1, 3, 32, 32], [55, 55], None, 'bicubic'],
        [[1, 3, 32, 32], None, 1.3, 'bicubic'],
        [[1, 3, 32, 32], [64, 63], None, 'bilinear'],
        [[1, 3, 32, 32], None, 1.3, 'bilinear'],
    ],
)
def test_upsample(input_size: List[int], size: Optional[Tuple[int, int]], scale_factor: Optional[float], mode: str):
    check_module(
        model=torch.nn.Upsample(size=size, scale_factor=scale_factor, mode=mode), args=[torch.randn(input_size)]
    )


if __name__ == '__main__':
    pytest.main([__file__])
