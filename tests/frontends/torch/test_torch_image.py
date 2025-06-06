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
from hidet.testing.torch_utils import check_module, FunctionalModule


@pytest.mark.parametrize('shape', [[2, 2]])
@pytest.mark.parametrize('normalized_shape', [2])
@pytest.mark.parametrize('dtype', [torch.float32])
def test_layer_norm(shape, normalized_shape, dtype, device):
    check_module(
        torch.nn.LayerNorm(normalized_shape=normalized_shape), [torch.randn(shape, dtype=dtype)], device=device
    )


@pytest.mark.parametrize('shape', [[1, 4, 32, 32]])
@pytest.mark.parametrize('num_groups', [1, 2, 4])
@pytest.mark.parametrize('dtype', [torch.float32])
def test_group_norm(shape, num_groups, dtype, device):
    check_module(
        torch.nn.GroupNorm(num_groups=num_groups, num_channels=shape[1]),
        [torch.randn(shape, dtype=dtype)],
        device=device,
    )


@pytest.mark.parametrize(
    'shape',
    [
        [1, 4, 32, 32],  # Standard image size
        [2, 8, 16, 16],  # Batch size 2, more channels
        [1, 16, 8, 8],  # More channels, smaller spatial dims
        [4, 12, 24, 24],  # Larger batch, non-power-2 channels
    ],
)
@pytest.mark.parametrize('num_groups', [1, 2, 4, 6])  # Added more group variations
@pytest.mark.parametrize('dtype', [torch.float32])
@pytest.mark.parametrize('eps', [1e-5])
def test_functional_group_norm(shape, num_groups, dtype, eps, device):
    # Skip invalid combinations where num_channels is not divisible by num_groups
    if shape[1] % num_groups != 0:
        pytest.skip(f"Skipping: {shape[1]} channels not divisible by {num_groups} groups")

    # Create random input
    x = torch.randn(shape, dtype=dtype, device=device)

    # Create random weight and bias
    num_channels = shape[1]
    weight = torch.randn(num_channels, dtype=dtype, device=device)
    bias = torch.randn(num_channels, dtype=dtype, device=device)

    # Test with weight and bias
    check_module(
        FunctionalModule(op=lambda x: torch.nn.functional.group_norm(x, num_groups, weight, bias, eps)),
        [x],
        device=device,
        atol=1e-4,
        rtol=1e-4,
    )

    # Test without weight and bias
    check_module(
        FunctionalModule(op=lambda x: torch.nn.functional.group_norm(x, num_groups)),
        [x],
        device=device,
        atol=1e-4,
        rtol=1e-4,
    )


@pytest.mark.parametrize(
    "input_size, size, scale_factor, mode, align_corners",
    [
        [[1, 3, 32, 32], (64, 64), None, 'nearest', None],
        [[1, 3, 32, 32], None, 1.3, 'nearest', None],
        [[1, 3, 32, 32], [55, 55], None, 'bicubic', False],
        [[1, 3, 32, 32], None, 1.3, 'bicubic', True],
        [[1, 3, 32, 32], [64, 63], None, 'bilinear', True],
        [[1, 3, 32, 32], None, 1.3, 'bilinear', False],
        [[1, 64, 256, 256], None, 1.0, 'bilinear', True],
        [[1, 64, 64, 64], None, 2.0, 'bilinear', True],
    ],
)
def test_upsample(
    input_size: List[int],
    size: Optional[Tuple[int, int]],
    scale_factor: Optional[float],
    mode: str,
    align_corners: Optional[bool],
    device,
):
    check_module(
        model=torch.nn.Upsample(size=size, scale_factor=scale_factor, mode=mode, align_corners=align_corners),
        args=[torch.randn(input_size)],
        device=device,
    )


if __name__ == '__main__':
    pytest.main([__file__])
