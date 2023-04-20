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
import torch.nn.functional
import hidet


@pytest.mark.parametrize("hidet_op", [hidet.ops.conv3d_transpose])
@pytest.mark.parametrize(
    'in_channels, out_channels, kernel_size, stride, pads, dilations, groups, depth, height, width, output_padding',
    [[10, 20, (5, 5, 5), (3, 2, 2), (2, 1, 1), (1, 1, 1), 5, 12, 11, 10, (1, 1, 1)]],
)
def test_conv3d_transpose(
    hidet_op,
    in_channels,
    out_channels,
    kernel_size,
    stride,
    pads,
    dilations,
    groups,
    depth,
    height,
    width,
    output_padding,
):
    torch_data = torch.ones(1, in_channels, depth, height, width, dtype=torch.float32).cuda()
    torch_weight = torch.ones(
        out_channels, in_channels // groups, kernel_size[0], kernel_size[1], kernel_size[2], dtype=torch.float32
    ).cuda()

    torch_output = torch.nn.functional.conv3d(
        torch_data, torch_weight, stride=stride, padding=pads, groups=groups, bias=None, dilation=1
    )
    hidet_data = hidet.from_torch(torch_data)
    hidet_weight = hidet.from_torch(torch_weight)
    hidet_output = hidet.ops.conv_pad(hidet_data, pads)
    hidet_output = hidet.ops.conv3d(hidet_output, hidet_weight, stride, dilations, groups=groups)
    np.testing.assert_allclose(hidet_output.cpu().numpy(), torch_output.cpu().numpy(), atol=1e-5)
    torch_transpose_output = torch.nn.functional.conv_transpose3d(
        torch_output,
        torch_weight,
        stride=stride,
        padding=pads,
        groups=groups,
        bias=None,
        dilation=1,
        output_padding=output_padding,
    )
    hidet_transpose_output = hidet_op(hidet_output, hidet_weight, stride, pads, groups, output_padding=output_padding)
    np.testing.assert_allclose(hidet_transpose_output.cpu().numpy(), torch_transpose_output.cpu().numpy(), atol=1e-5)


if __name__ == '__main__':
    pytest.main([__file__])
