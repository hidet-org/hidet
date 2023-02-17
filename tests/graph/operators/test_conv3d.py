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
from hidet.testing import check_torch_binary


@pytest.mark.parametrize("hidet_op", [ops.conv3d, ops.conv3d_gemm])
@pytest.mark.parametrize(
    "n, c, d, h, w, oc, kz, kx, ky",
    [
        [1, 3, 32, 32, 32, 12, 3, 3, 3],  # kernel 3,
        [2, 3, 32, 32, 32, 12, 7, 7, 7],  # kernel 7, batch size 2
        [1, 3, 32, 32, 32, 12, 1, 1, 1],  # kernel 1,
    ],
)
@pytest.mark.parametrize("padding", [[0, 0, 0, 0, 0, 0], [1, 2, 3, 1, 2, 3]])
@pytest.mark.parametrize("stride", [[1, 1, 1], [3, 2, 1]])
@pytest.mark.parametrize("dilations", [[1, 1, 1], [1, 2, 3]])
def test_conv3d(hidet_op, n, c, d, h, w, oc, kz, kx, ky, padding, stride, dilations):
    check_torch_binary(
        a_shape=[n, c, d, h, w],
        b_shape=[oc, c, kz, kx, ky],
        torch_func=lambda data, weight: torch.nn.functional.conv3d(
            data, weight, bias=None, stride=stride, padding=[padding[0], padding[1], padding[2]], dilation=dilations
        ),
        hidet_func=lambda data, weight: hidet_op(
            ops.conv_pad(data, padding), weight, stride=stride, dilations=dilations
        ),
        dtype='float32',
        atol=2e-1,
        rtol=2e-1,
    )


if __name__ == '__main__':
    pytest.main([__file__])
