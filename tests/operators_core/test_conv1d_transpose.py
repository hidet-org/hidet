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
import torch
import hidet
from hidet.testing import check_binary, check_binary_dynamic


def torch_conv_transpose1d(
    data: np.ndarray, weight: np.ndarray, padding: int, stride: int, output_padding: int, groups: int
):
    data_torch, weight_torch = torch.from_numpy(data), torch.from_numpy(weight)
    torch_out = torch.nn.functional.conv_transpose1d(
        data_torch,
        weight_torch,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        groups=groups,
        bias=None,
    )
    return torch_out.numpy()


@pytest.mark.parametrize(
    'in_channels, out_channels, kernel_size, stride, pads, groups, length, output_padding',
    [[10, 20, 5, 2, 0, 5, 15, 1]],
)
def test_conv1d_transpose(in_channels, out_channels, kernel_size, stride, pads, groups, length, output_padding):
    check_binary(
        a_shape=[1, out_channels, length],
        b_shape=[out_channels, in_channels // groups, kernel_size],
        numpy_op=lambda data, weight: torch_conv_transpose1d(data, weight, pads, stride, output_padding, groups),
        hidet_op=lambda data, weight: hidet.ops.conv1d_transpose(data, weight, stride, pads, groups, output_padding),
        dtype='float32',
        atol=2e-5,
        rtol=2e-5,
    )


@pytest.mark.parametrize(
    'in_channels, out_channels, kernel_size, stride, pads, groups, length, output_padding',
    [[10, 20, 5, 2, 0, 5, 15, 1]],
)
def test_conv1d_transpose_dynamic(in_channels, out_channels, kernel_size, stride, pads, groups, length, output_padding):
    check_binary_dynamic(
        a_shape=[('b', 1), ('oc', out_channels), ('l', length)],
        b_shape=[out_channels, in_channels // groups, kernel_size],
        numpy_op=lambda data, weight: torch_conv_transpose1d(data, weight, pads, stride, output_padding, groups),
        hidet_op=lambda data, weight: hidet.ops.conv1d_transpose(data, weight, stride, pads, groups, output_padding),
        dtype='float32',
        atol=2e-5,
        rtol=2e-5,
    )


if __name__ == '__main__':
    pytest.main([__file__])
