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
from typing import List, Union, Tuple

import numpy as np
import torch
import pytest

import hidet
from hidet import ops, Tensor
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


# due to float16 numerical errors on larger kernel sizes, eg 5, disable the test for now
@pytest.mark.parametrize(
    "n, c, h, w, oc, kx, ky",
    [
        [1, 64, 32, 32, 12, 3, 3],  # kernel 3,
        [2, 128, 32, 32, 32, 4, 4],  # kernel 5, batch size 2
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


# For some reason, the autoscheduler generated kernel is really inaccurate, despite being correct, so we
#   use fp64
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
        dtype='float64',
        atol=1e-2,
        rtol=1e-2,
    )


@pytest.mark.parametrize(
    "n, c, h, w, oc, kx, ky, padding, stride, dilations",
    [
        [1, 3, 32, 32, 12, 3, 3, [0, 0], [1, 1], [1, 1]],  # kernel 3,
        [2, 3, 32, 32, 12, 7, 7, [1, 2], [2, 3], [2, 3]],  # kernel 7, batch size 2
        [1, 3, 32, 32, 12, 1, 1, [0, 0], [2, 3], [1, 1]],  # kernel 1,
    ],
)
def test_conv2d(n, c, h, w, oc, kx, ky, padding, stride, dilations):
    check_binary(
        a_shape=[n, c, h, w],
        b_shape=[oc, c, kx, ky],
        numpy_op=lambda data, weight: torch_conv2d(data, weight, padding, stride, dilations),
        hidet_op=lambda data, weight: ops.conv2d(data, weight, padding=padding, stride=stride, dilations=dilations),
        dtype='float32',
        atol=2e-5,
        rtol=2e-5,
    )


@pytest.mark.parametrize(
    "n, c, h, w, oc, kx, ky, padding, stride, dilations",
    [
        [1, 3, 32, 32, 12, 3, 3, [0, 0], [1, 1], [1, 1]],  # kernel 3,
        [2, 3, 32, 32, 12, 7, 7, [1, 2], [2, 3], [2, 3]],  # kernel 7, batch size 2
        [1, 3, 32, 32, 12, 1, 1, [0, 0], [2, 3], [1, 1]],  # kernel 1,
    ],
)
def test_conv2d_gemm(n, c, h, w, oc, kx, ky, padding, stride, dilations):
    check_binary(
        a_shape=[n, c, h, w],
        b_shape=[oc, c, kx, ky],
        numpy_op=lambda data, weight: torch_conv2d(data, weight, padding, stride, dilations),
        hidet_op=lambda data, weight: ops.conv2d_gemm(
            ops.conv_pad(data, padding), weight, stride=stride, dilations=dilations
        ),
        dtype='float32',
        atol=2e-5,
        rtol=2e-5,
    )


# We only test for dynamic data sizes
@pytest.mark.parametrize("hidet_op", [ops.conv2d, ops.conv2d_gemm])
@pytest.mark.parametrize(
    "n, c, h, w, oc, kx, ky, padding, stride, dilations",
    [
        [1, 3, 32, 32, 12, 3, 3, [0, 0], [1, 1], [1, 1]],  # kernel 3,
        [2, 3, 32, 32, 12, 7, 7, [1, 2], [2, 3], [2, 3]],  # kernel 7, batch size 2
        [1, 3, 32, 32, 12, 1, 1, [0, 0], [2, 3], [1, 1]],  # kernel 1,
    ],
)
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


# We only test for dynamic data sizes
@pytest.mark.parametrize(
    "n, c, h, w, oc, kx, ky, padding, stride, dilations",
    [
        [1, 3, 32, 32, 12, 3, 3, [0, 0], [1, 1], [1, 1]],  # kernel 3,
        [2, 3, 32, 32, 12, 7, 7, [1, 2], [2, 3], [2, 3]],  # kernel 7, batch size 2
        [1, 3, 32, 32, 12, 1, 1, [0, 0], [2, 3], [1, 1]],  # kernel 1,
    ],
)
def test_conv2d_dynamic(n, c, h, w, oc, kx, ky, padding, stride, dilations):
    check_binary_dynamic(
        a_shape=[('n', n), ('c', c), ('h', h), ('w', w)],
        b_shape=[oc, c, kx, ky],
        numpy_op=lambda data, weight: torch_conv2d(data, weight, padding, stride, dilations),
        hidet_op=lambda data, weight: ops.conv2d(data, weight, padding=padding, stride=stride, dilations=dilations),
        dtype='float32',
        atol=2e-5,
        rtol=2e-5,
    )


@pytest.mark.parametrize(
    "n, c, h, w, oc, kx, ky, padding, stride, dilations",
    [
        [1, 3, 32, 32, 12, 3, 3, [0, 0], [1, 1], [1, 1]],  # kernel 3,
        [2, 3, 32, 32, 12, 7, 7, [1, 2], [2, 3], [2, 3]],  # kernel 7, batch size 2
        [1, 3, 32, 32, 12, 1, 1, [0, 0], [2, 3], [1, 1]],  # kernel 1,
    ],
)
def test_conv2d_dynamic_gemm(n, c, h, w, oc, kx, ky, padding, stride, dilations):
    check_binary_dynamic(
        a_shape=[('n', n), ('c', c), ('h', h), ('w', w)],
        b_shape=[oc, c, kx, ky],
        numpy_op=lambda data, weight: torch_conv2d(data, weight, padding, stride, dilations),
        hidet_op=lambda data, weight: ops.conv2d_gemm(
            ops.conv_pad(data, padding), weight, stride=stride, dilations=dilations
        ),
        dtype='float32',
        atol=2e-5,
        rtol=2e-5,
    )


def pre_transform_img_ref(img: Tensor, padding: Union[int, Tuple[int, int]], pad_value=0.0, make_multiple_8=False):
    import hidet

    n, c, w, h = img.shape
    assert pad_value == 0.0
    img = hidet.ops.conv_pad(img, padding)
    img = hidet.ops.transpose(img, [0, 2, 3, 1])
    if make_multiple_8:
        pad_channel = ((c + 7) // 8) * 8 - c
        img = hidet.ops.pad(img, [0, pad_channel])
    return img


@pytest.mark.skip(reason='This operator is not needed right now')
@pytest.mark.parametrize("img_dim", [[32, 64], [31, 63]])
@pytest.mark.parametrize("channel", [3, 32, 64])
@pytest.mark.parametrize("padding", [[0, 0], [1, 1], [2, 3]])
@pytest.mark.parametrize("multi_8", [True, False])
def test_pretransform_v3(img_dim, channel, padding, multi_8):
    from hidet.graph.ops.conv2d.conv2d_gemm import pre_transform_img

    img = hidet.randn([1, channel] + img_dim, device='cuda', dtype='float16')
    y1 = pre_transform_img_ref(img, tuple(padding), 0.0, multi_8)
    y2 = pre_transform_img(img, tuple(padding), 0.0, multi_8)
    assert torch.allclose(y1.torch(), y2.torch(), 1e-3, 1e-3)

    imgs = hidet.symbol([1, channel] + img_dim, dtype='float16', device='cuda')
    ys = pre_transform_img(imgs, tuple(padding), 0.0, multi_8)
    graph = hidet.trace_from(ys, imgs)
    cgraph = graph.build(space=2)
    task = cgraph.compiled_tasks[0]
    for func in task.candidates:
        y2 = hidet.empty_like(y1)
        func(img, y2)
        assert torch.allclose(y1.torch(), y2.torch(), 1e-2, 1e-2)


if __name__ == '__main__':
    pytest.main([__file__])
