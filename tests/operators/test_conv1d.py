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
from typing import List

import numpy as np
import torch
import pytest

from hidet import ops
from hidet.testing import check_binary


def torch_conv1d(
    data: np.ndarray, weight: np.ndarray, padding: List[int], stride: List[int], dilations: List[int], groups: int
):
    data_torch, weight_torch = torch.from_numpy(data), torch.from_numpy(weight)
    torch_out = torch.nn.functional.conv1d(
        data_torch, weight_torch, bias=None, stride=stride, padding=[padding[0]], dilation=dilations, groups=groups
    )
    return torch_out.numpy()


@pytest.mark.parametrize("hidet_op", [ops.conv1d])
@pytest.mark.parametrize("n, c, l, oc, k", [[1, 3, 32, 12, 3]])
@pytest.mark.parametrize("padding", [[0], [1]])
@pytest.mark.parametrize("stride", [1])
@pytest.mark.parametrize("dilations", [1])
@pytest.mark.parametrize("groups", [1])
def test_conv1d(hidet_op, n, c, l, oc, k, padding, stride, dilations, groups):
    check_binary(
        a_shape=[n, c, l],
        b_shape=[oc, c, k],
        numpy_op=lambda data, weight: torch_conv1d(data, weight, padding, stride, dilations, groups),
        hidet_op=lambda data, weight: hidet_op(
            ops.conv_pad(data, padding), weight=weight, stride=stride, dilations=dilations, groups=groups
        ),
        dtype='float32',
        atol=2e-5,
        rtol=2e-5,
    )


if __name__ == '__main__':
    pytest.main([__file__])
