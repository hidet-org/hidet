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
"""
Winograd convolution, see <Fast Algorithms for Convolutional Neural Networks> https://arxiv.org/pdf/1509.09308.pdf
"""
from functools import lru_cache
from typing import List, Tuple

import numpy as np

from hidet.ir.expr import const_tensor, Constant, cast
from hidet.graph.ops.definitions.matmul import matmul
from ..utils import Tensor, Operator, Task, TensorNode, input_like, compute, reduce, normalize_kernel


@lru_cache(maxsize=32)
def winograd_transform_matrices(m: int, r: int) -> Tuple[Constant, Constant, Constant]:
    if m == 2 and r == 3:
        G = np.array([[1, 0, 0], [1 / 2, 1 / 2, 1 / 2], [1 / 2, -1 / 2, 1 / 2], [0, 0, 1]]).astype(np.float32)
        BT = np.array([[1, 0, -1, 0], [0, 1, 1, 0], [0, -1, 1, 0], [0, 1, 0, -1]]).astype(np.float32)
        AT = np.array([[1, 1, 1, 0], [0, 1, -1, -1]]).astype(np.float32)
        return const_tensor(G), const_tensor(BT), const_tensor(AT)
    else:
        raise NotImplementedError('winograd transform matrices: m = {}, r = {}'.format(m, r))


class Conv2dWinogradImageTransformTask(Task):
    def __init__(self, x: TensorNode, kernel: List[int], ms: List[int]):
        # pylint: disable=too-many-locals
        assert len(kernel) == 2 and len(x.const_shape()) == 4
        n, c, h, w = x.const_shape()
        rx, ry = kernel
        mx, my = ms  # output size per tile
        oh, ow = h - rx + 1, w - ry + 1  # output size of image
        nh, nw = (oh + mx - 1) // mx, (ow + my - 1) // my  # number of tiles on each image dimension
        p = n * nh * nw  # number of tiles per channel
        alpha_x, alpha_y = mx + rx - 1, my + ry - 1
        tile = compute(
            name='tile',
            shape=[c, p, alpha_x, alpha_y],
            fcompute=lambda cc, pp, ax, ay: x[pp // (nh * nw), cc, (pp // nw) % nh * mx + ax, pp % nw * my + ay],
        )
        BH = winograd_transform_matrices(mx, rx)[1]
        BW = winograd_transform_matrices(my, ry)[1]
        dtype = x.ttype.dtype
        y = compute(
            name='y',
            shape=[alpha_x, alpha_y, c, p],
            fcompute=lambda ax, ay, cc, pp: reduce(
                shape=[alpha_x, alpha_y],
                fcompute=lambda kx, ky: cast(BH[ax, kx], dtype) * tile[cc, pp, kx, ky] * cast(BW[ay, ky], dtype),
                reduce_type='sum',
            ),
        )
        super().__init__(name='conv2d_winograd_image_transform', inputs=[x], outputs=[y])


class Conv2dWinogradFilterTransformTask(Task):
    def __init__(self, w: TensorNode, ms: List[int]):
        assert len(w.const_shape()) == 4
        oc, c, rx, ry = w.const_shape()
        mx, my = ms
        alpha_x, alpha_y = mx + rx - 1, my + ry - 1
        GH = winograd_transform_matrices(mx, rx)[0]
        GW = winograd_transform_matrices(my, ry)[0]
        dtype = w.ttype.dtype
        y = compute(
            name='y',
            shape=[alpha_x, alpha_y, oc, c],
            fcompute=lambda ax, ay, occ, cc: reduce(
                shape=[rx, ry],
                fcompute=lambda kx, ky: cast(GH[ax, kx], dtype) * w[occ, cc, kx, ky] * cast(GW[ay, ky], dtype),
                reduce_type='sum',
            ),
        )
        super().__init__(name='conv2d_winograd_filter_transform', inputs=[w], outputs=[y])


class Conv2dWinogradInverseTransformTask(Task):
    def __init__(self, y: TensorNode, input_shape, kernel, ms):
        assert len(y.const_shape()) == 4
        alpha_x, alpha_y, oc, p = y.const_shape()
        n, c, h, w = input_shape  # pylint: disable=unused-variable
        rx, ry = kernel
        mx, my = ms
        oh, ow = h - rx + 1, w - ry + 1  # output size of image
        nh, nw = (oh + mx - 1) // mx, (ow + my - 1) // my  # number of tiles on each image dimension
        AH = winograd_transform_matrices(mx, rx)[2]
        AW = winograd_transform_matrices(my, ry)[2]
        dtype = y.ttype.dtype
        inverse = compute(
            name='inverse',
            shape=[mx, my, oc, p],
            fcompute=lambda mxx, myy, occ, pp: reduce(
                shape=[alpha_x, alpha_y],
                fcompute=lambda kx, ky: cast(AH[mxx, kx], dtype) * y[kx, ky, occ, pp] * cast(AW[myy, ky], dtype),
                reduce_type='sum',
            ),
        )
        output = compute(
            name='output',
            shape=[n, oc, oh, ow],
            fcompute=lambda nn, occ, ohh, oww: inverse[
                ohh % mx, oww % my, occ, nn * (nh * nw) + (ohh // mx) * nw + (oww // my)
            ],
        )
        super().__init__(name='conv2d_winograd_inverse_transform', inputs=[y], outputs=[output])


class Conv2dWinogradImageTransformOp(Operator):
    def __init__(self, x: Tensor, kernel, ms):
        if len(x.shape) != 4:
            raise NotImplementedError('Current only support winograd conv2d')
        kernel = normalize_kernel(kernel, dim=2)
        assert len(ms) == 2
        super().__init__(
            inputs=[x],
            task=Conv2dWinogradImageTransformTask(input_like(x, 'x'), kernel, ms),
            attributes={'kernel': kernel, 'ms': ms},
        )


class Conv2dWinogradFilterTransformOp(Operator):
    def __init__(self, w: Tensor, ms):
        assert len(ms) == 2
        super().__init__(
            inputs=[w], task=Conv2dWinogradFilterTransformTask(input_like(w, 'w'), ms), attributes={'ms': ms}
        )


class Conv2dWinogradInverseTransformOp(Operator):
    def __init__(self, y: Tensor, input_shape, kernel, ms):
        kernel = normalize_kernel(kernel, dim=2)
        super().__init__(
            inputs=[y],
            task=Conv2dWinogradInverseTransformTask(input_like(y, 'y'), input_shape, kernel, ms),
            attributes={'input_shape': input_shape, 'kernel': kernel, 'ms': ms},
        )


def conv2d_winograd_image_transform(x: Tensor, kernel, ms) -> Tensor:
    return Conv2dWinogradImageTransformOp(x, kernel, ms).get_output(0)


def conv2d_winograd_filter_transform(w: Tensor, ms) -> Tensor:
    return Conv2dWinogradFilterTransformOp(w, ms).get_output(0)


def conv2d_winograd_inverse_transform(y: Tensor, input_shape, kernel, ms) -> Tensor:
    return Conv2dWinogradInverseTransformOp(y, input_shape, kernel, ms).get_output(0)


def conv2d_winograd(x: Tensor, w: Tensor) -> Tensor:
    assert len(x.shape) == 4 and len(w.shape) == 4 and x.shape[1] == w.shape[1]
    r2m = {1: 1, 3: 2}
    for k in w.shape[2:]:
        if k not in r2m:
            raise NotImplementedError('Winograd convolution for kernel size {} has not been supported yet.'.format(k))

    input_shape = x.shape
    kernel = w.shape[2:]
    ms = [r2m[r] for r in kernel]

    # winograd transform
    x = conv2d_winograd_image_transform(x, kernel, ms)  # [alpha_x, alpha_y, ci, p]
    w = conv2d_winograd_filter_transform(w, ms)  # [alpha_x, alpha_y, co, ci]

    # product
    y = matmul(w, x)  # [alpha_x * alpha_y, co, p]

    # winograd inverse transform
    y = conv2d_winograd_inverse_transform(y, input_shape, kernel, ms)  # [n, oc, oh, ow]
    return y
