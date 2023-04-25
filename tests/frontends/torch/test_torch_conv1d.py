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


@pytest.mark.parametrize('in_shape,w_shape,stride,padding,groups', [[[1, 3, 224], [42, 3, 7], 2, 1, 1]])
@pytest.mark.parametrize('dtype', [torch.float32])
def test_conv1d(in_shape, w_shape, stride, padding, groups, dtype):
    cudnn.allow_tf32 = False
    check_module(
        model=torch.nn.Conv1d(
            in_channels=in_shape[1],
            out_channels=w_shape[0],
            kernel_size=w_shape[2:],
            stride=stride,
            padding=padding,
            groups=groups,
        ),
        args=[torch.randn(in_shape, dtype=dtype)],
    )
    cudnn.allow_tf32 = True


if __name__ == '__main__':
    pytest.main([__file__])
