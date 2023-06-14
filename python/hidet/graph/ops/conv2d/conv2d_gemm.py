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
from typing import Any, Dict, List, Optional, Sequence
from hidet.graph.tensor import Tensor

from hidet.ir.expr import is_constant
from hidet.graph.ops.matmul import matmul, batch_matmul
from hidet.graph.ops.utils import Task, Operator, Tensor, compute, input_like, TensorNode
from hidet.graph.ops.utils import normalize_kernel, normalize_stride, normalize_dilations, reduce
from hidet.ir.task import Task
from .utils import infer_conv2d_shape


class Conv2dGemmImageTransformTask(Task):
    def __init__(self, x: TensorNode, kernel: List[int], stride: List[int], dilations: List[int], groups: int):
        n, c, h, w = x.shape
        kx, ky = kernel
        sx, sy = stride
        dilx, dily = dilations
        p, q = (h - dilx * (kx - 1) - 1) // sx + 1, (w - dily * (ky - 1) - 1) // sy + 1
        if is_constant(c) and c % groups != 0:
            msg = 'Conv2d expect in_channels % groups == 0, but got in_channels {} and groups {}'.format(c, groups)
            raise ValueError(msg)
        gc = c // groups  # group channels
        gemm_x = compute(
            name='gemm_x',
            shape=[groups, n * p * q, gc * kx * ky],
            fcompute=lambda g, i, k: x[
                i // (p * q), g * gc + k // (kx * ky), i // q % p * sx + k // ky % kx * dilx, i % q * sy + k % ky * dily
            ],
        )
        super().__init__(name='conv2d_gemm_image_transform', inputs=[x], outputs=[gemm_x])


class Conv2dGemmImageTransformOp(Operator):
    def __init__(self, x: Tensor, kernel, stride, dilations, groups):
        kernel = normalize_kernel(kernel)
        stride = normalize_stride(stride)
        super().__init__(
            inputs=[x],
            attributes={'kernel': kernel, 'stride': stride, 'groups': groups, 'dilations': dilations},
            task=Conv2dGemmImageTransformTask(input_like(x, 'x'), kernel, stride, dilations, groups),
        )


class Conv2dGemmPostTransformTask(Task):
    def __init__(self, a: TensorNode, stride: List[int], dilations: List[int]):
        n, ky, kx, oc, h, w = a.shape
        stride_y, stride_x = stride
        dilation_y, dilation_x = dilations
        out_h = (h - dilation_y * (ky - 1) - 1) // stride_y + 1
        out_w = (w - dilation_x * (kx - 1) - 1) // stride_x + 1
        out = compute(
            name='post_transform',
            shape=[n, oc, out_h, out_w], 
            fcompute=lambda ni, oci, hi, wi: reduce(
                shape=[ky, kx],
                reduce_type='sum',
                fcompute=lambda kyi, kxi: a[ni, kyi, kxi, oci, hi * stride_y + kyi * dilation_y, wi * stride_x + kxi * dilation_x]
            )
        )
        super().__init__(name="conv2d_gemm_post_transform", inputs=[a], outputs=[out])


class Conv2dGemmPostTransformOp(Operator):
    def __init__(self, a: Tensor, stride: List[int], dilations: List[int]):
        stride = normalize_stride(stride)
        dilations = normalize_dilations(dilations)
        super().__init__(
            inputs=[a],
            attributes={'stride': stride, 'dilations': dilations},
            task=Conv2dGemmPostTransformTask(input_like(a, 'a'), stride, dilations),
        )


def conv2d_gemm_image_transform(
    x: Tensor, kernel: Sequence[int], stride: Sequence[int], dilations: Sequence[int], groups: int = 1
) -> Tensor:
    return Conv2dGemmImageTransformOp(x, kernel, stride, dilations, groups).get_output(0)


def conv2d_gemm_post_transform(
    a: Tensor, stride: Sequence[int], dilations: Sequence[int]
) -> Tensor:
    """
    Equivalent to the following algorithm:

    def post_transform(a, stride, dilations):
        n, ky, kx, oc, h, w = a.shape
        stride_y = stride[0]
        stride_x = stride[1]
        dilation_y = dilations[0]
        dilation_x = dilations[1]
        out_h = (h - dilation_y * (ky - 1) - 1) // stride_y + 1
        out_w = (w - dilation_x * (kx - 1) - 1) // stride_x + 1
        y = torch.zeros([n, oc, out_h, out_w], device=x.device, dtype=x.dtype)

        for i in range(out_h):
            for j in range(out_w):
                for kyi in range(ky):
                    for kxi in range(kx):
                        iy = i * stride_y + kyi * dilation_y
                        ix = j * stride_x + kxi * dilation_x
                        y[:, :, i, j] += a[:, kyi, kxi, :, iy, ix]
        return y
    """
    return Conv2dGemmPostTransformOp(a, stride, dilations).get_output(0)


def conv2d_gemm_filter_transform(w: Tensor, groups: int = 1) -> Tensor:
    # weight shape: [oc, c, kx, ky]
    # output shape: [groups, c * kx * ky, ogc] where ogc = oc // groups
    oc, c, kx, ky = w.shape
    if is_constant(oc, groups) and oc % groups != 0:
        raise ValueError('invalid conv2d groups {} for out channels {}'.format(groups, oc))
    ogc = oc // groups
    w = w.reshape([groups, ogc, c, kx, ky])  # [groups, ogc, c, kx, ky]
    w = w.rearrange([[0], [2, 3, 4], [1]])  # [groups, c * kx * ky, ogc]
    return w


def conv2d_gemm_inverse_transform(gemm_y: Tensor, out_height, out_width) -> Tensor:
    # gemm_y shape: [groups, n * p * q, ogc]
    # output shape: [n, oc, p, q] where oc = groups * ogc
    p, q = out_height, out_width
    groups, npq, ogc = gemm_y.shape
    if is_constant(npq, p, q) and npq % (p * q) != 0:
        raise ValueError('invalid conv2d output shape {} for height {} and width {}'.format(npq, p, q))
    n = npq // (p * q)
    y = gemm_y.reshape([groups, n, p, q, ogc])
    y = y.rearrange([[1], [0, 4], [2], [3]])
    return y


def conv2d_gemm(data: Tensor, weight: Tensor, stride, dilations: List[int], groups: int = 1) -> Tensor:
    gemm_x = conv2d_gemm_image_transform(
        data, kernel=weight.shape[2:], stride=stride, dilations=dilations, groups=groups
    )
    gemm_w = conv2d_gemm_filter_transform(weight, groups=groups)
    gemm_y = matmul(gemm_x, gemm_w)

    y_shape = infer_conv2d_shape(data.shape, weight.shape, stride, groups, dilations)
    y = conv2d_gemm_inverse_transform(gemm_y, out_height=y_shape[2], out_width=y_shape[3])
    return y


def conv2d_gemm_1(data: Tensor, gemm_w: Tensor, weight_shape, stride, dilations: List[int], groups: int = 1) -> Tensor:
    gemm_x = conv2d_gemm_image_transform(
        data, kernel=weight_shape[2:], stride=stride, dilations=dilations, groups=groups
    )
    
    gemm_y = matmul(gemm_x, gemm_w)

    y_shape = infer_conv2d_shape(data.shape, weight_shape, stride, groups, dilations)
    y = conv2d_gemm_inverse_transform(gemm_y, out_height=y_shape[2], out_width=y_shape[3])
    return y


def conv2d_gemm_fp16(data: Tensor, weight: Tensor, stride, dilations: List[int], groups: int = 1, mma='simt') -> Tensor:
    gemm_x = conv2d_gemm_image_transform(
        data, kernel=weight.shape[2:], stride=stride, dilations=dilations, groups=groups
    )
    gemm_w = conv2d_gemm_filter_transform(weight, groups=groups)
    gemm_y = batch_matmul(gemm_x, gemm_w, mma=mma)

    y_shape = infer_conv2d_shape(data.shape, weight.shape, stride, groups, dilations)
    y = conv2d_gemm_inverse_transform(gemm_y, out_height=y_shape[2], out_width=y_shape[3])
    return y


def conv2d_gemm_2(data: Tensor, weight: Tensor, stride, dilations: List[int], groups: int = 1) -> Tensor:
    from hidet.graph.ops import transpose
    stride = normalize_stride(stride)
    dilations = normalize_dilations(dilations)
    
    n, c, h, w = data.shape
    oc, wc, ky, kx = weight.shape
    data = data.reshape([n, 1, groups, wc, w * h])
    weight = transpose(weight, [2, 3, 0, 1]).reshape([ky * kx, groups, oc // groups, wc])

    a = matmul(weight, data).reshape([n, ky, kx, oc, h, w])
    y = conv2d_gemm_post_transform(a, stride, dilations)
    return y


def conv2d_gemm_2_1(data: Tensor, weight: Tensor, weight_shape: List[int], stride, dilations: List[int], groups: int = 1) -> Tensor:
    stride = normalize_stride(stride)
    dilations = normalize_dilations(dilations)
    
    n, c, h, w = data.shape
    oc, wc, ky, kx = weight_shape
    data = data.reshape([n, 1, groups, wc, w * h])

    a = matmul(weight, data).reshape([n, ky, kx, oc, h, w])
    y = conv2d_gemm_post_transform(a, stride, dilations)
    return y
