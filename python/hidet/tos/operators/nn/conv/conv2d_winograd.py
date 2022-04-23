from typing import List, Sequence, Union, Optional, Tuple
from functools import lru_cache
import numpy as np
from hidet.ir.expr import const_tensor, Constant
from hidet.tos.operators.common import Tensor, Operator, Task, TensorInput, input_like, compute, reduce, Grid, normalize_padding, normalize_kernel
from hidet.tos.operators.nn.matmul import batched_matmul
from hidet.tos.operators.basic.transform import flatten, reshape

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


def winograd_image_transform_task(x: TensorInput, padding: List[int], kernel: List[int], ms: List[int]) -> Task:
    assert len(kernel) == 2 and len(padding) == 4 and len(x.const_shape()) == 4
    rx, ry = kernel
    n, c, h, w = x.const_shape()
    mx, my = ms                                         # output size per tile
    oh, ow = h - rx + 1, w - ry + 1                     # output size of image
    nh, nw = (oh + mx - 1) // mx, (ow + my - 1) // my   # number of tiles on each image dimension
    p = n * nh * nw                                     # number of tiles per channel
    alpha_x, alpha_y = mx + rx - 1, my + ry - 1

    pad = compute(
        name='pad',
        shape=[n, c, h + padding[0] + padding[2], w + padding[1] + padding[3]],
        fcompute=lambda nn, cc, hh, ww: x.protect_read([nn, cc, hh - padding[0], ww - padding[1]], default_value=0.0),
    )
    tile = compute(
        name='tile',
        shape=[c, p, alpha_x, alpha_y],
        fcompute=lambda cc, pp, ax, ay: pad[p // (nh * nw), c, (p // nw) % nh * mx + ax, p % nh * my + ay]
    )
    BH = winograd_transform_matrices(mx, rx)[1]
    BW = winograd_transform_matrices(my, ry)[1]
    y = compute(
        name='y',
        shape=[alpha_x, alpha_y, c, p],
        fcompute=lambda ax, ay, cc, pp: reduce(
            shape=[alpha_x, alpha_y],
            fcompute=lambda kx, ky: BH[ax, kx] * tile[c, p, kx, ky] * BW[ay, ky],
            reduce_type='sum'
        )
    )
    task = Task(
        name='winograd_image_transform',
        computation=y,
        params=[x, y],
        worker=Grid()
    )
    return task


def winograd_filter_transform_task(w: TensorInput, ms: List[int]) -> Task:
    assert len(w.const_shape()) == 4
    oc, c, rx, ry = w.const_shape()
    mx, my = ms
    alpha_x, alpha_y = mx + rx - 1, my + ry - 1
    GH = winograd_transform_matrices(mx, rx)[1]
    GW = winograd_transform_matrices(my, ry)[1]
    y = compute(
        name='y',
        shape=[alpha_x, alpha_y, oc, c],
        fcompute=lambda ax, ay, occ, cc: reduce(
            shape=[rx, ry],
            fcompute=lambda kx, ky: GH[ax, rx] * w[occ, cc, rx, ry] * GW[ky, ay],
            reduce_type='sum'
        )
    )
    task = Task(
        name='winograd_filter_transform',
        computation=y,
        params=[w, y],
        worker=Grid()
    )
    return task


def winograd_inverse_transform_task(y: TensorInput, input_shape, padding, kernel, ms) -> Task:
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
            fcompute=lambda kx, ky: AH[mx, kx] * y[kx, ky, occ, pp] * AW[ky, my],
            reduce_type='sum'
        )
    )
    output = compute(
        name='output',
        shape=[n, oc, oh, ow],
        fcompute=lambda nn, occ, ohh, oww: inverse[ohh % mx, oww % my, occ, nn * (ohh // mx) * (oww // my)],
    )
    task = Task(
        name='winograd_inverse_transform',
        computation=output,
        params=[y, output],
        worker=Grid()
    )
    return task


class WinogradImageTransformOp(Operator):
    def __init__(self, x: Tensor, padding, kernel, ms):
        if len(x.shape) != 4:
            raise NotImplementedError('Current only support winograd conv2d')
        padding = normalize_padding(padding, dim=2)
        kernel = normalize_kernel(kernel, dim=2)
        assert len(ms) == 2
        super().__init__(
            inputs=[x],
            task=winograd_image_transform_task(input_like(x, 'x'), padding, kernel, ms),
            padding=padding,
            kernel=kernel,
            ms=ms
        )


class WinogradFilterTransformOp(Operator):
    def __init__(self, w: Tensor, ms):
        assert len(ms) == 2
        super().__init__(
            inputs=[w],
            task=winograd_filter_transform_task(input_like(w, 'w'), ms),
            ms=ms
        )


class WinogradInverseTransformOp(Operator):
    def __init__(self, y: Tensor, input_shape, padding, kernel, ms):
        padding = normalize_padding(padding, dim=2)
        kernel = normalize_kernel(kernel, dim=2)
        super().__init__(
            inputs=[y],
            task=winograd_inverse_transform_task(input_like(y, 'y'), input_shape, padding, kernel, ms),
            input_shape=input_shape,
            padding=padding,
            kernel=kernel,
            ms=ms
        )


def winograd_image_transform(x: Tensor, padding, kernel, ms) -> Tensor:
    return WinogradImageTransformOp(x, padding, kernel, ms).get_output(0)


def winograd_filter_transform(w: Tensor, ms) -> Tensor:
    return WinogradFilterTransformOp(w, ms).get_output(0)


def winograd_inverse_transform(y: Tensor, input_shape, padding, kernel, ms) -> Tensor:
    return WinogradInverseTransformOp(y, input_shape, padding, kernel, ms).get_output(0)


def conv2d_winograd(x: Tensor, w: Tensor, padding) -> Tensor:
    assert len(x.shape) == 4
    r2m = {
        1: 1,
        3: 2
    }
    input_shape = x.shape
    kernel = w.shape[2:]
    ms = [r2m[r] for r in kernel]
    alpha = [r + m - 1 for r, m in zip(kernel, ms)]

    # winograd transform
    x = winograd_image_transform(x, padding, kernel, ms)  # [alpha_x, alpha_y, ci, p]
    w = winograd_filter_transform(w, ms)                  # [alpha_x, alpha_y, co, ci]

    # product
    x = flatten(x, start_dim=0, end_dim=2)              # [alpha_x * alpha_y, ci, p]
    w = flatten(w, start_dim=0, end_dim=2)              # [alpha_x * alpha_y, co, ci]
    y = batched_matmul(w, x)                            # [alpha_x * alpha_y, co, p]
    y = reshape(y, [alpha[0], alpha[1], 0, 0])          # [alpha_x, alpha_y, co, p]

    # winograd inverse transform
    y = winograd_inverse_transform(y, input_shape, padding, kernel, ms)  # [n, oc, oh, ow]
    return y
