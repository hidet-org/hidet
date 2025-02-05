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
import torch.nn.functional
import hidet
from hidet.testing import check_torch_binary, check_torch_binary_dynamic


@pytest.mark.parametrize("hidet_op", [hidet.ops.conv3d_transpose])
@pytest.mark.parametrize(
    'in_channels, out_channels, kernel_size, stride, pads, groups, depth, height, width, output_padding',
    [[10, 20, (5, 5, 5), (3, 2, 2), (2, 1, 1), 5, 12, 11, 10, (1, 1, 1)]],
)
def test_conv3d_transpose(
    hidet_op, in_channels, out_channels, kernel_size, stride, pads, groups, depth, height, width, output_padding
):
    torch_transpose_fn = lambda torch_output, torch_weight: torch.nn.functional.conv_transpose3d(
        torch_output,
        torch_weight,
        stride=stride,
        padding=pads,
        groups=groups,
        bias=None,
        dilation=1,
        output_padding=output_padding,
    )

    hidet_transpose_fn = lambda hidet_output, hidet_weight: hidet_op(
        hidet_output, hidet_weight, stride, pads, groups, output_padding=output_padding
    )

    check_torch_binary(
        a_shape=[1, out_channels, depth, height, width],
        b_shape=[out_channels, in_channels // groups, kernel_size[0], kernel_size[1], kernel_size[2]],
        torch_func=torch_transpose_fn,
        hidet_func=hidet_transpose_fn,
        dtype='float32',
        atol=1e-5,
        rtol=2e-1,
    )


@pytest.mark.parametrize("hidet_op", [hidet.ops.conv3d_transpose])
@pytest.mark.parametrize(
    'in_channels, out_channels, kernel_size, stride, pads, groups, depth, height, width, output_padding',
    [[10, 20, (5, 5, 5), (3, 2, 2), (2, 1, 1), 5, 12, 11, 10, (1, 1, 1)]],
)
def test_conv3d_transpose_dynamic(
    hidet_op, in_channels, out_channels, kernel_size, stride, pads, groups, depth, height, width, output_padding
):
    torch_transpose_fn = lambda torch_output, torch_weight: torch.nn.functional.conv_transpose3d(
        torch_output,
        torch_weight,
        stride=stride,
        padding=pads,
        groups=groups,
        bias=None,
        dilation=1,
        output_padding=output_padding,
    )

    hidet_transpose_fn = lambda hidet_output, hidet_weight: hidet_op(
        hidet_output, hidet_weight, stride, pads, groups, output_padding=output_padding
    )

    check_torch_binary_dynamic(
        a_shape=[('b', 1), ('oc', out_channels), ('d', depth), ('h', height), ('w', width)],
        b_shape=[out_channels, in_channels // groups, kernel_size[0], kernel_size[1], kernel_size[2]],
        torch_func=torch_transpose_fn,
        hidet_func=hidet_transpose_fn,
        dtype='float32',
        atol=1e-5,
        rtol=2e-1,
    )


if __name__ == '__main__':
    pytest.main([__file__])
