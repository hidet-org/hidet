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
from hidet.graph.ops.utils import Operator, input_like
from hidet.graph.ops.utils import normalize_kernel, normalize_stride
from hidet.graph.tensor import Tensor
from hidet.ir.compute import TensorNode
from hidet.ir.compute import compute
from hidet.ir.expr import is_constant
from hidet.ir.task import Task
from .utils import infer_conv1d_shape


class Conv1dGemmImageTransformTask(Task):
    def __init__(self, x: TensorNode, kernel: int, stride: int, dilation: int, groups: int):
        n, c, h = x.shape
        kx = kernel
        sx = stride
        dilx = dilation
        p = (h - dilx * (kx - 1) - 1) // sx + 1
        self._assert(
            c % groups == 0,
            msg='Conv1d expect in_channels % groups == 0, but got in_channels {} and groups {}'.format(c, groups),
        )
        gc = c // groups  # group channels
        gemm_x = compute(
            name='gemm_x',
            shape=[groups, n * p, gc * kx],
            fcompute=lambda g, i, k: x[i // p, g * gc + k // kx, i % p * sx + k % kx * dilx],
        )
        super().__init__(name='conv1d_gemm_image_transform', inputs=[x], outputs=[gemm_x])


class Conv1dGemmImageTransformOp(Operator):
    def __init__(self, x: Tensor, kernel, stride, dilations, groups):
        (kernel,) = normalize_kernel(kernel, dim=1)
        (stride,) = normalize_stride(stride, dim=1)
        super().__init__(
            inputs=[x],
            attributes={'kernel': kernel, 'stride': stride, 'groups': groups, 'dilations': dilations},
            task=Conv1dGemmImageTransformTask(input_like(x, 'x'), kernel, stride, dilations, groups),
        )


def conv1d_gemm_image_transform(x: Tensor, kernel: int, stride: int, dilation: int, groups: int = 1) -> Tensor:
    return Conv1dGemmImageTransformOp(x, kernel, stride, dilation, groups).outputs[0]


def conv1d_gemm_filter_transform(w: Tensor, groups: int = 1) -> Tensor:
    # weight shape: [oc, c, kx]
    # output shape: [groups, c * kx, ogc] where ogc = oc // groups
    oc, c, kx = w.shape
    # TODO: current assertion mechanism does not cover this use case (only on the task-level)
    if is_constant(oc, groups) and oc % groups != 0:
        raise ValueError('invalid conv1d groups {} for out channels {}'.format(groups, oc))
    ogc = oc // groups
    w = w.reshape([groups, ogc, c, kx])  # [groups, ogc, c, kx]
    w = w.rearrange([[0], [2, 3], [1]])  # [groups, c * kx, ogc]
    return w


def conv1d_gemm_inverse_transform(gemm_y: Tensor, out_height) -> Tensor:
    # gemm_y shape: [groups, n * p, ogc]
    # output shape: [n, oc, p] where oc = groups * ogc
    p = out_height
    groups, npq, ogc = gemm_y.shape
    # TODO: current assertion mechanism does not cover this use case (only on the task-level)
    if is_constant(npq, p) and npq % p != 0:
        raise ValueError('invalid conv1d output shape {} for dimension {}'.format(npq, p))
    n = npq // p
    y = gemm_y.reshape([groups, n, p, ogc])
    y = y.rearrange([[1], [0, 3], [2]])
    return y


def conv1d_gemm(data: Tensor, weight: Tensor, stride, dilation: int = 1, groups: int = 1) -> Tensor:
    from hidet import ops

    gemm_x = conv1d_gemm_image_transform(data, kernel=weight.shape[2], stride=stride, dilation=dilation, groups=groups)
    gemm_w = conv1d_gemm_filter_transform(weight, groups=groups)
    gemm_y = ops.matmul(gemm_x, gemm_w, require_prologue=True)

    y_shape = infer_conv1d_shape(data.shape, weight.shape, stride, groups, dilation)
    y = conv1d_gemm_inverse_transform(gemm_y, out_height=y_shape[2])
    return y
