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

from hidet.graph.ops.definitions.matmul import matmul
from hidet.graph.ops.definitions.utils import Task, Operator, Tensor, compute, input_like, TensorNode
from hidet.graph.ops.definitions.utils import normalize_kernel, normalize_stride
from .utils import infer_conv3d_shape


class Conv3dGemmImageTransformTask(Task):
    def __init__(self, x: TensorNode, kernel: List[int], stride: List[int], dilations: List[int], groups: int):
        n, c, d, h, w = x.const_shape()
        kz, kx, ky = kernel
        sz, sx, sy = stride
        dilz, dilx, dily = dilations
        r, p, q = (
            (d - dilz * (kz - 1) - 1) // sz + 1,
            (h - dilx * (kx - 1) - 1) // sx + 1,
            (w - dily * (ky - 1) - 1) // sy + 1,
        )
        if c % groups != 0:
            msg = 'Conv3d expect in_channels % groups == 0, but got in_channels {} and groups {}'.format(c, groups)
            raise ValueError(msg)
        gc = c // groups  # group channels
        gemm_x = compute(
            name='gemm_x',
            shape=[groups, n * r * p * q, gc * kz * kx * ky],
            fcompute=lambda g, i, k: x[
                i // (r * p * q),
                g * gc + k // (kz * kx * ky),
                i // (p * q) % r * sz + k // (kx * ky) % kz * dilz,
                i // q % p * sx + k // ky % kx * dilx,
                i % q * sy + k % ky * dily,
            ],
        )
        super().__init__(name='conv2d_gemm_image_transform', inputs=[x], outputs=[gemm_x])


class Conv3dGemmImageTransformOp(Operator):
    def __init__(self, x: Tensor, kernel, stride, dilations, groups):
        kernel = normalize_kernel(kernel, dim=3)
        stride = normalize_stride(stride, dim=3)
        super().__init__(
            inputs=[x],
            task=Conv3dGemmImageTransformTask(input_like(x, 'x'), kernel, stride, dilations, groups),
            attributes={'kernel': kernel, 'stride': stride, 'groups': groups, 'dilations': dilations},
        )


def conv3d_gemm_image_transform(
    x: Tensor, kernel: List[int], stride: List[int], dilations: List[int], groups: int = 1
) -> Tensor:
    return Conv3dGemmImageTransformOp(x, kernel, stride, dilations, groups).get_output(0)


def conv3d_gemm_filter_transform(w: Tensor, groups: int = 1) -> Tensor:
    # weight shape: [oc, c, kz * kx, ky]
    # output shape: [groups, c * kz * kx * ky, ogc] where ogc = oc // groups
    oc, c, kz, kx, ky = w.shape
    if oc % groups != 0:
        raise ValueError('invalid conv3d groups {} for out channels {}'.format(groups, oc))
    ogc = oc // groups
    w = w.reshape([groups, ogc, c, kz, kx, ky])  # [groups, ogc, c, kz, kx, ky]
    w = w.rearrange([[0], [2, 3, 4, 5], [1]])  # [groups, c * kz * kx * ky, ogc]
    return w


def conv3d_gemm_inverse_transform(gemm_y: Tensor, out_depth, out_height, out_width) -> Tensor:
    # gemm_y shape: [groups, n * r * p * q, ogc]
    # output shape: [n, oc, r, p, q] where oc = groups * ogc
    r, p, q = out_depth, out_height, out_width
    groups, nrpq, ogc = gemm_y.shape
    assert nrpq % (r * p * q) == 0
    n = nrpq // (r * p * q)
    y = gemm_y.reshape([groups, n, r, p, q, ogc])
    y = y.rearrange([[1], [0, 5], [2], [3], [4]])
    return y


def conv3d_gemm(data: Tensor, weight: Tensor, stride, dilations: List[int], groups: int = 1) -> Tensor:
    gemm_x = conv3d_gemm_image_transform(
        data, kernel=weight.shape[2:], stride=stride, dilations=dilations, groups=groups
    )
    gemm_w = conv3d_gemm_filter_transform(weight, groups=groups)
    gemm_y = matmul(gemm_x, gemm_w)

    y_shape = infer_conv3d_shape(data.shape, weight.shape, stride, groups, dilations)
    y = conv3d_gemm_inverse_transform(gemm_y, out_depth=y_shape[2], out_height=y_shape[3], out_width=y_shape[4])
    return y
