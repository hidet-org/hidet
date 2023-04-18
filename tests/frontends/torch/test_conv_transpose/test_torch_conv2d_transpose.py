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
import torch.backends.cudnn as cudnn


@pytest.mark.parametrize('in_channels', [3])
@pytest.mark.parametrize('out_channels', [64])
@pytest.mark.parametrize('kernel_size', [(3, 5)])
@pytest.mark.parametrize('stride', [(3, 2)])
@pytest.mark.parametrize('padding', [(2, 1)])
@pytest.mark.parametrize('output_padding', [(2, 1)])
@pytest.mark.parametrize('groups', [1])
@pytest.mark.parametrize('dilation', [1])
@pytest.mark.parametrize('dtype', [torch.float32])
def test_conv2d_transpose(
    in_channels, out_channels, kernel_size, stride, padding, output_padding, groups, dilation, dtype
):
    print(in_channels, out_channels, kernel_size, stride, padding, output_padding, groups, dilation)
    check_module(
        model=torch.nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            bias=False,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            dilation=dilation,
        ),
        args=[torch.randn([1, 3, 224, 224], dtype=dtype)],
        atol=2e-4,
    )
    cudnn.allow_tf32 = True


if __name__ == '__main__':
    pytest.main([__file__])
