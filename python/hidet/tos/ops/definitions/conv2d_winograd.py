from functools import lru_cache
from typing import List, Tuple

import numpy as np

from hidet.ir.expr import const_tensor, Constant
from .matmul import batched_matmul
from .transform import flatten, reshape
from .utils import Tensor, Operator, Task, TensorNode, input_like, compute, reduce, normalize_padding, normalize_kernel

"""
Winograd convolution, see <Fast Algorithms for Convolutional Neural Networks> https://arxiv.org/pdf/1509.09308.pdf
"""


@lru_cache(maxsize=32)
def winograd_transform_matrices(m: int, r: int) -> Tuple[Constant, Constant, Constant]:
    if m == 2 and r == 3:
        G = np.array(
            [[1, 0, 0],
             [1/2, 1/2, 1/2],
             [1/2, -1/2, 1/2],
             [0, 0, 1]]
        ).astype(np.float32)
        BT = np.array(
            [[1, 0, -1, 0],
             [0, 1, 1, 0],
             [0, -1, 1, 0],
             [0, 1, 0, -1]]
        ).astype(np.float32)
        AT = np.array(
            [[1, 1, 1, 0],
             [0, 1, -1, -1]]
        ).astype(np.float32)
        return const_tensor(G), const_tensor(BT), const_tensor(AT)
    else:
        raise NotImplementedError('winograd transform matrices: m = {}, r = {}'.format(m, r))


class Conv2dWinogradImageTransformTask(Task):
    def __init__(self, x: TensorNode, padding: List[int], kernel: List[int], ms: List[int]):
        assert len(kernel) == 2 and len(padding) == 4 and len(x.const_shape()) == 4
        n, c, h, w = x.const_shape()

        pad = compute(
            name='pad',
            shape=[n, c, h + padding[0] + padding[2], w + padding[1] + padding[3]],
            fcompute=lambda nn, cc, hh, ww: x.protect_read([nn, cc, hh - padding[0], ww - padding[1]], default_value=0.0),
        )

        n, c, h, w = pad.const_shape()
        rx, ry = kernel
        mx, my = ms                                         # output size per tile
        oh, ow = h - rx + 1, w - ry + 1                     # output size of image
        nh, nw = (oh + mx - 1) // mx, (ow + my - 1) // my   # number of tiles on each image dimension
        p = n * nh * nw                                     # number of tiles per channel
        alpha_x, alpha_y = mx + rx - 1, my + ry - 1
        tile = compute(
            name='tile',
            shape=[c, p, alpha_x, alpha_y],
            fcompute=lambda cc, pp, ax, ay: pad[pp // (nh * nw), cc, (pp // nw) % nh * mx + ax, pp % nw * my + ay]
        )
        BH = winograd_transform_matrices(mx, rx)[1]
        BW = winograd_transform_matrices(my, ry)[1]
        y = compute(
            name='y',
            shape=[alpha_x, alpha_y, c, p],
            fcompute=lambda ax, ay, cc, pp: reduce(
                shape=[alpha_x, alpha_y],
                fcompute=lambda kx, ky: BH[ax, kx] * tile[cc, pp, kx, ky] * BW[ay, ky],
                reduce_type='sum'
            )
        )
        super().__init__(
            name='conv2d_winograd_image_transform',
            inputs=[x],
            outputs=[y]
        )


class Conv2dWinogradFilterTransformTask(Task):
    def __init__(self, w: TensorNode, ms: List[int]):
        assert len(w.const_shape()) == 4
        oc, c, rx, ry = w.const_shape()
        mx, my = ms
        alpha_x, alpha_y = mx + rx - 1, my + ry - 1
        GH = winograd_transform_matrices(mx, rx)[0]
        GW = winograd_transform_matrices(my, ry)[0]
        y = compute(
            name='y',
            shape=[alpha_x, alpha_y, oc, c],
            fcompute=lambda ax, ay, occ, cc: reduce(
                shape=[rx, ry],
                fcompute=lambda kx, ky: GH[ax, kx] * w[occ, cc, kx, ky] * GW[ay, ky],
                reduce_type='sum'
            )
        )
        super().__init__(
            name='conv2d_winograd_filter_transform',
            inputs=[w],
            outputs=[y]
        )


class Conv2dWinogradInverseTransform(Task):
    def __init__(self, y: TensorNode, input_shape, padding, kernel, ms):
        assert len(y.const_shape()) == 4
        alpha_x, alpha_y, oc, p = y.const_shape()
        n, c, h, w = input_shape
        h, w = h + padding[0] + padding[2], w + padding[1] + padding[3]
        rx, ry = kernel
        mx, my = ms
        oh, ow = h - rx + 1, w - ry + 1                     # output size of image
        nh, nw = (oh + mx - 1) // mx, (ow + my - 1) // my   # number of tiles on each image dimension
        AH = winograd_transform_matrices(mx, rx)[2]
        AW = winograd_transform_matrices(my, ry)[2]
        inverse = compute(
            name='inverse',
            shape=[mx, my, oc, p],
            fcompute=lambda mxx, myy, occ, pp: reduce(
                shape=[alpha_x, alpha_y],
                fcompute=lambda kx, ky: AH[mxx, kx] * y[kx, ky, occ, pp] * AW[myy, ky],
                reduce_type='sum'
            )
        )
        output = compute(
            name='output',
            shape=[n, oc, oh, ow],
            fcompute=lambda nn, occ, ohh, oww: inverse[ohh % mx, oww % my, occ, nn * (nh * nw) + (ohh // mx) * nw + (oww // my)],
        )
        super().__init__(
            name='conv2d_winograd_inverse_transform',
            inputs=[y],
            outputs=[output]
        )


class WinogradImageTransformOp(Operator):
    def __init__(self, x: Tensor, padding, kernel, ms):
        if len(x.shape) != 4:
            raise NotImplementedError('Current only support winograd conv2d')
        padding = normalize_padding(padding, dim=2)
        kernel = normalize_kernel(kernel, dim=2)
        assert len(ms) == 2
        super().__init__(
            inputs=[x],
            task=Conv2dWinogradImageTransformTask(input_like(x, 'x'), padding, kernel, ms),
            padding=padding,
            kernel=kernel,
            ms=ms
        )


class WinogradFilterTransformOp(Operator):
    def __init__(self, w: Tensor, ms):
        assert len(ms) == 2
        super().__init__(
            inputs=[w],
            task=Conv2dWinogradFilterTransformTask(input_like(w, 'w'), ms),
            ms=ms
        )


class WinogradInverseTransformOp(Operator):
    def __init__(self, y: Tensor, input_shape, padding, kernel, ms):
        padding = normalize_padding(padding, dim=2)
        kernel = normalize_kernel(kernel, dim=2)
        super().__init__(
            inputs=[y],
            task=Conv2dWinogradInverseTransform(input_like(y, 'y'), input_shape, padding, kernel, ms),
            input_shape=input_shape,
            padding=padding,
            kernel=kernel,
            ms=ms
        )


def conv2d_winograd_image_transform(x: Tensor, padding, kernel, ms) -> Tensor:
    return WinogradImageTransformOp(x, padding, kernel, ms).get_output(0)


def conv2d_winograd_filter_transform(w: Tensor, ms) -> Tensor:
    return WinogradFilterTransformOp(w, ms).get_output(0)


def conv2d_winograd_inverse_transform(y: Tensor, input_shape, padding, kernel, ms) -> Tensor:
    return WinogradInverseTransformOp(y, input_shape, padding, kernel, ms).get_output(0)


def conv2d_winograd(x: Tensor, w: Tensor, padding) -> Tensor:
    assert len(x.shape) == 4 and len(w.shape) == 4 and x.shape[1] == w.shape[1]
    r2m = {
        1: 1,
        3: 2
    }
    for k in w.shape[2:]:
        if k not in r2m:
            raise NotImplementedError('Winograd convolution for kernel size {} has not been supported yet.'.format(k))

    input_shape = x.shape
    kernel = w.shape[2:]
    ms = [r2m[r] for r in kernel]
    alpha = [r + m - 1 for r, m in zip(kernel, ms)]

    # winograd transform
    x = conv2d_winograd_image_transform(x, padding, kernel, ms)  # [alpha_x, alpha_y, ci, p]
    w = conv2d_winograd_filter_transform(w, ms)                  # [alpha_x, alpha_y, co, ci]

    # product
    x = flatten(x, start_dim=0, end_dim=2)                          # [alpha_x * alpha_y, ci, p]
    w = flatten(w, start_dim=0, end_dim=2)                          # [alpha_x * alpha_y, co, ci]
    y = batched_matmul(w, x)                                        # [alpha_x * alpha_y, co, p]
    y = reshape(y, [alpha[0], alpha[1], y.shape[1], y.shape[2]])    # [alpha_x, alpha_y, co, p]

    # winograd inverse transform
    y = conv2d_winograd_inverse_transform(y, input_shape, padding, kernel, ms)  # [n, oc, oh, ow]
    return y
