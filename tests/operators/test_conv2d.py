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
from hidet.testing import check_binary, check_binary_dynamic, check_torch_binary


def torch_conv2d(
    data: np.ndarray, weight: np.ndarray, padding: List[int], stride: List[int], dilations: List[int], groups: int = 1
):
    data_torch, weight_torch = torch.from_numpy(data), torch.from_numpy(weight)
    needs_convert = False
    if data_torch.dtype == torch.float16 and not data_torch.is_cuda:
        data_torch = data_torch.cuda()
        weight_torch = weight_torch.cuda()
        needs_convert = True
    torch_out = torch.nn.functional.conv2d(
        data_torch,
        weight_torch,
        bias=None,
        stride=stride,
        padding=[padding[0], padding[1]],
        dilation=dilations,
        groups=groups,
    )
    if needs_convert:
        torch_out = torch_out.cpu()
    return torch_out.numpy()


@pytest.mark.parametrize(
    "n, c, h, w, oc, kx, ky",
    [
        [1, 64, 32, 32, 12, 3, 3],  # kernel 3,
        [2, 128, 32, 32, 32, 5, 5],  # kernel 7, batch size 2
        [1, 32, 32, 32, 64, 1, 1],  # kernel 1,
    ],
)
@pytest.mark.parametrize("groups", [1, 2, 4])
@pytest.mark.parametrize("stride", [[1, 1], [2, 3]])
@pytest.mark.parametrize("dilations", [[1, 1], [2, 3]])
@pytest.mark.parametrize("parallel_k", [1, 2, 3])
@pytest.mark.parametrize(
    "device", ["cuda"]
)  # we don't test for cpu because its quite imprecise in fp16 for larger kernel sizes
def test_conv2d_gemm_fp16(n, c, h, w, oc, kx, ky, groups, stride, dilations, parallel_k, device):
    tol = 0.8
    check_binary(
        a_shape=[n, c, h, w],
        b_shape=[oc, c // groups, kx, ky],
        numpy_op=lambda data, weight: torch_conv2d(data, weight, [0, 0], stride, dilations, groups),
        hidet_op=lambda data, weight: ops.transpose(
            ops.conv2d_gemm_fp16_channel_last(
                ops.transpose(data, [0, 2, 3, 1]),
                weight,
                stride=stride,
                dilations=dilations,
                groups=groups,
                parallel_k_parts=parallel_k,
            ),
            [0, 3, 1, 2],
        ),
        dtype='float16',
        device=device,
        atol=tol,
        rtol=tol,
    )


@pytest.mark.parametrize(
    "n, c, h, w, oc, kx, ky",
    [
        [1, 64, 32, 32, 12, 3, 3],  # kernel 3,
        [2, 128, 32, 32, 32, 5, 5],  # kernel 7, batch size 2
        [1, 32, 32, 32, 64, 1, 1],  # kernel 1,
    ],
)
@pytest.mark.parametrize("groups", [1, 2, 4])
@pytest.mark.parametrize("stride", [[1, 1], [2, 3]])
@pytest.mark.parametrize("dilations", [[1, 1], [2, 3]])
def test_conv2d_channel_last(n, c, h, w, oc, kx, ky, groups, stride, dilations):
    check_torch_binary(
        a_shape=[n, c, h, w],
        b_shape=[oc, c // groups, kx, ky],
        torch_func=lambda data, weight: torch.nn.functional.conv2d(
            data, weight, bias=None, stride=stride, padding=[0, 0], dilation=dilations, groups=groups
        ),
        hidet_func=lambda data, weight: ops.transpose(
            ops.conv2d_channel_last(
                ops.transpose(data, [0, 2, 3, 1]), weight, stride=stride, dilations=dilations, groups=groups
            ),
            [0, 3, 1, 2],
        ),
        atol=0.5,
        rtol=0.5,
    )


@pytest.mark.parametrize("hidet_op", [ops.conv2d, ops.conv2d_gemm])
@pytest.mark.parametrize(
    "n, c, h, w, oc, kx, ky",
    [
        [1, 3, 32, 32, 12, 3, 3],  # kernel 3,
        [2, 3, 32, 32, 12, 7, 7],  # kernel 7, batch size 2
        [1, 3, 32, 32, 12, 1, 1],  # kernel 1,
    ],
)
@pytest.mark.parametrize("padding", [[0, 0, 0, 0], [1, 2, 1, 2]])
@pytest.mark.parametrize("stride", [[1, 1], [2, 3]])
@pytest.mark.parametrize("dilations", [[1, 1], [2, 3]])
def test_conv2d(hidet_op, n, c, h, w, oc, kx, ky, padding, stride, dilations):
    check_binary(
        a_shape=[n, c, h, w],
        b_shape=[oc, c, kx, ky],
        numpy_op=lambda data, weight: torch_conv2d(data, weight, padding, stride, dilations),
        hidet_op=lambda data, weight: hidet_op(ops.conv_pad(data, padding), weight, stride=stride, dilations=dilations),
        dtype='float32',
        atol=2e-5,
        rtol=2e-5,
    )


# We only test for dynamic data sizes
@pytest.mark.parametrize("hidet_op", [ops.conv2d, ops.conv2d_gemm])
@pytest.mark.parametrize(
    "n, c, h, w, oc, kx, ky",
    [
        [1, 3, 32, 32, 12, 3, 3],  # kernel 3,
        [2, 3, 32, 32, 12, 7, 7],  # kernel 7, batch size 2
        [1, 3, 32, 32, 12, 1, 1],  # kernel 1,
    ],
)
@pytest.mark.parametrize("padding", [[0, 0, 0, 0], [1, 2, 1, 2]])
@pytest.mark.parametrize("stride", [[1, 1], [2, 3]])
@pytest.mark.parametrize("dilations", [[1, 1], [2, 3]])
def test_conv2d_dynamic(hidet_op, n, c, h, w, oc, kx, ky, padding, stride, dilations):
    check_binary_dynamic(
        a_shape=[('n', n), ('c', c), ('h', h), ('w', w)],
        b_shape=[oc, c, kx, ky],
        numpy_op=lambda data, weight: torch_conv2d(data, weight, padding, stride, dilations),
        hidet_op=lambda data, weight: hidet_op(ops.conv_pad(data, padding), weight, stride=stride, dilations=dilations),
        dtype='float32',
        atol=2e-5,
        rtol=2e-5,
    )


if __name__ == '__main__':
    pytest.main([__file__])
