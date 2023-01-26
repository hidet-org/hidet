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
from typing import Tuple

import numpy as np
import pytest
import torch.nn.functional

from hidet import ops
from hidet.testing import check_unary, check_torch_unary


def numpy_pool2d(
    data: np.ndarray, kernel: Tuple[int, int], stride: Tuple[int, int], padding: Tuple[int, int, int, int], reduce_type
) -> np.ndarray:
    assert reduce_type in ['max', 'avg']
    n, c, h, w = data.shape
    kx, ky = kernel
    sx, sy = stride
    ph, pw = h + padding[0] + padding[2], w + padding[1] + padding[3]
    padded = np.full_like(data, fill_value=0.0 if reduce_type == 'avg' else -1e30, shape=(n, c, ph, pw))
    padded[:, :, padding[0] : padding[0] + h, padding[1] : padding[1] + w] = data
    oh, ow = (ph - kx) // sx + 1, (pw - ky) // sy + 1
    output = np.empty_like(data, shape=(n, c, oh, ow))
    for nn in range(n):
        for cc in range(c):
            for p in range(oh):
                for q in range(ow):
                    if reduce_type == 'max':
                        output[nn, cc, p, q] = np.max(padded[nn, cc, p * sx : p * sx + kx, q * sy : q * sy + ky])
                    elif reduce_type == 'avg':
                        output[nn, cc, p, q] = np.sum(padded[nn, cc, p * sx : p * sx + kx, q * sy : q * sy + ky]) / (
                            kx * ky
                        )

    return output


@pytest.mark.parametrize(
    "shape, kernel, stride, padding",
    [
        [[1, 1, 1, 1], [3, 3], [1, 1], [1, 1, 1, 1]],  # kernel 3, stride 1
        [[1, 3, 32, 32], [3, 3], [1, 1], [1, 1, 1, 1]],  # kernel 3, stride 1
        [[1, 3, 32, 32], [3, 3], [2, 2], [1, 1, 1, 1]],  # kernel 3, stride 2
        [[1, 3, 32, 32], [7, 7], [1, 1], [3, 3, 3, 3]],  # kernel 3, stride 1
        [[1, 3, 32, 32], [7, 7], [2, 2], [3, 3, 3, 3]],  # kernel 3, stride 2
    ],
)
def test_max_pool2d(shape, kernel, stride, padding):
    check_unary(
        shape,
        lambda x: numpy_pool2d(x, kernel, stride, padding, 'max'),
        lambda x: ops.max_pool2d(x, kernel, stride, padding),
        atol=0,
        rtol=0,
    )


@pytest.mark.parametrize(
    "shape, kernel, stride, padding",
    [
        [[1, 1, 1, 1, 1], [3, 3, 3], [1, 1, 1], [1, 1, 1, 1, 1, 1]],  # kernel 3, stride 1
        [[1, 3, 32, 32, 32], [3, 3, 3], [1, 1, 1], [1, 1, 1, 1, 1, 1]],  # kernel 3, stride 1
        [[1, 3, 32, 32, 32], [3, 3, 3], [2, 2, 2], [1, 1, 1, 1, 1, 1]],  # kernel 3, stride 2
        [[1, 3, 32, 32, 32], [7, 7, 7], [1, 1, 1], [3, 3, 3, 3, 3, 3]],  # kernel 3, stride 1
        [[1, 3, 32, 32, 32], [7, 7, 7], [2, 2, 2], [3, 3, 3, 3, 3, 3]],  # kernel 3, stride 2
    ],
)
def test_max_pool3d(shape, kernel, stride, padding):
    check_torch_unary(
        shape,
        lambda x: torch.nn.functional.max_pool3d(
            x, kernel_size=kernel, stride=stride, padding=[padding[0], padding[1], padding[2]]
        ),
        lambda x: ops.max_pool3d(x, kernel, stride, padding),
        atol=0,
        rtol=0,
    )


@pytest.mark.parametrize(
    "shape, kernel, stride, padding",
    [
        [[1, 1, 1, 1], [3, 3], [1, 1], [1, 1, 1, 1]],  # kernel 3, stride 1
        [[1, 3, 32, 32], [3, 3], [1, 1], [1, 1, 1, 1]],  # kernel 3, stride 1
        [[1, 3, 32, 32], [3, 3], [2, 2], [1, 1, 1, 1]],  # kernel 3, stride 2
        [[1, 3, 32, 32], [7, 7], [1, 1], [3, 3, 3, 3]],  # kernel 3, stride 1
        [[1, 3, 32, 32], [7, 7], [2, 2], [3, 3, 3, 3]],  # kernel 3, stride 2
    ],
)
def test_avg_pool2d(shape, kernel, stride, padding):
    check_unary(
        shape,
        lambda x: numpy_pool2d(x, kernel, stride, padding, 'avg'),
        lambda x: ops.avg_pool2d(x, kernel, stride, padding),
        atol=1e-5,
        rtol=1e-5,
    )


@pytest.mark.parametrize(
    "shape, kernel, stride, padding",
    [
        [[1, 3, 32, 32, 32], [3, 3, 3], [1, 1, 1], [1, 1, 1, 1, 1, 1]],  # kernel 3, stride 1
        [[1, 3, 32, 32, 32], [3, 3, 3], [2, 2, 2], [1, 1, 1, 1, 1, 1]],  # kernel 3, stride 2
        [[1, 3, 32, 32, 32], [7, 7, 7], [1, 1, 1], [3, 3, 3, 3, 3, 3]],  # kernel 3, stride 1
        [[1, 3, 32, 32, 32], [7, 7, 7], [2, 2, 2], [3, 3, 3, 3, 3, 3]],  # kernel 3, stride 2
    ],
)
def test_avg_pool3d(shape, kernel, stride, padding):
    check_torch_unary(
        shape,
        lambda x: torch.nn.functional.avg_pool3d(
            x, kernel_size=kernel, stride=stride, padding=[padding[0], padding[1], padding[2]]
        ),
        lambda x: ops.avg_pool3d(x, kernel, stride, padding),
        atol=0,
        rtol=0,
    )


@pytest.mark.parametrize('reduce_type', ['max', 'avg'])
@pytest.mark.parametrize(
    'shape,output_size',
    [
        ((1, 3, 224, 224), (7, 7)),
        ((1, 3, 32, 32), (3, 3)),
        ((1, 2, 37, 63), (3, 2)),
        ((1, 1, 1, 1), (1, 1)),
        ((1, 3, 224), 7),
        ((1, 3, 32, 32, 32), (7, 7, 7)),
    ],
)
def test_adaptive_pool(shape, output_size, reduce_type):
    spatial_dim = len(shape) - 2
    torch_func = {
        (1, 'max'): torch.nn.functional.adaptive_max_pool1d,
        (1, 'avg'): torch.nn.functional.adaptive_avg_pool1d,
        (2, 'max'): torch.nn.functional.adaptive_max_pool2d,
        (2, 'avg'): torch.nn.functional.adaptive_avg_pool2d,
        (3, 'max'): torch.nn.functional.adaptive_max_pool3d,
        (3, 'avg'): torch.nn.functional.adaptive_avg_pool3d,
    }[(spatial_dim, reduce_type)]
    hidet_func = {
        (1, 'max'): ops.adaptive_max_pool1d,
        (1, 'avg'): ops.adaptive_avg_pool1d,
        (2, 'max'): ops.adaptive_max_pool2d,
        (2, 'avg'): ops.adaptive_avg_pool2d,
        (3, 'max'): ops.adaptive_max_pool3d,
        (3, 'avg'): ops.adaptive_avg_pool3d,
    }[(spatial_dim, reduce_type)]
    check_torch_unary(
        shape,
        torch_func=lambda x: torch_func(x, output_size),
        hidet_func=lambda x: hidet_func(x, output_size),
        atol=1e-6,
        rtol=1e-6,
    )


if __name__ == '__main__':
    pytest.main([__file__])
