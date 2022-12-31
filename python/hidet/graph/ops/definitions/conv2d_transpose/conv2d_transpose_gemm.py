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
from typing import Sequence, Union, Tuple
from hidet.ir.expr import if_then_else, LogicalAnd
from hidet.graph.ops.definitions.utils import Task, Operator, Tensor, TensorNode
from hidet.graph.ops.definitions.utils import input_like, normalize_stride, normalize_padding
from hidet.ir.compute import compute
from hidet.graph.ops.definitions.matmul import matmul


class Conv2dTransposeGemmImageTask(Task):
    def __init__(
        self,
        data: TensorNode,
        kernel: Tuple[int, int],
        stride: Sequence[int],  # [sx, sy]
        padding: Sequence[int],  # [px0, py0, px1, py1]
        groups: int,
        output_padding: Sequence[int],  # [opx, opy]
    ):
        n, oc, p, q = data.const_shape()
        kx, ky = kernel
        sx, sy = stride
        px0, py0, px1, py1 = padding
        h = (p - 1) * sx + -px0 - px1 + kx + output_padding[0]
        w = (q - 1) * sy + -py0 - py1 + ky + output_padding[1]
        og = oc // groups  # output channels in each group

        def fcompute(b, i, k):
            gi = b
            ni, hi, wi = i // (h * w), ((i // w) % h), (i % w)
            ogi, kxi, kyi = k // (kx * ky), ((k // ky) % kx), (k % ky)
            xx = hi + px0 - kxi
            yy = wi + py0 - kyi
            return if_then_else(
                cond=LogicalAnd.join(xx >= 0, xx < p * sx, xx % sx == 0, yy >= 0, yy < q * sy, yy % sy == 0),
                then_expr=data[ni, gi * og + ogi, xx // sx, yy // sy],
                else_expr=0.0,
            )

        output = compute(name='gemm_x', shape=[groups, n * h * w, og * kx * ky], fcompute=fcompute)
        super().__init__(name='conv2d_transpose_gemm_image', inputs=[data], outputs=[output])


class Conv2dTransposeGemmImageOp(Operator):
    def __init__(
        self,
        data: Tensor,
        kernel: Tuple[int, int],
        stride: Tuple[int, int],
        padding: Tuple[int, int, int, int],
        groups: int,
        output_padding: Tuple[int, int],
    ):
        super().__init__(
            inputs=[data],
            task=Conv2dTransposeGemmImageTask(
                input_like(data, 'data'), kernel, stride, padding, groups, output_padding
            ),
            attributes={
                'kernel': kernel,
                'stride': stride,
                'padding': padding,
                'groups': groups,
                'output_padding': output_padding,
            },
        )


def conv2d_transpose_gemm_image(
    data: Tensor,
    kernel: Tuple[int, int],
    stride: Tuple[int, int],
    padding: Tuple[int, int, int, int],
    groups: int,
    output_padding: Tuple[int, int],
):
    # input shape: [n, oc, p, q]
    # output shape: [groups, n * h * w, og * kx * ky]
    return Conv2dTransposeGemmImageOp(data, kernel, stride, padding, groups, output_padding).get_output(0)


def conv2d_transpose_gemm_filter(weight: Tensor, groups: int = 1):
    # input shape: [oc, wc, kx, ky] where oc = groups * og
    # output shape: [groups, og * kx * ky, wc]
    oc, wc, kx, ky = weight.shape
    og = oc // groups
    return weight.reshape([groups, og, wc, kx, ky]).rearrange([[0], [1, 3, 4], [2]])


def conv2d_transpose_gemm_inverse(gemm_y, height: int, width: int):
    # input shape: [groups, n * h * w, wc]
    # output shape: [n, c, h, w] where c = groups * wc
    groups, nhw, wc = gemm_y.shape
    assert nhw % (height * width) == 0
    n = nhw // (height * width)
    return gemm_y.reshape([groups, n, height, width, wc]).rearrange([[1], [0, 4], [2], [3]])


def conv2d_transpose_gemm(
    data: Tensor,
    weight: Tensor,
    stride: Union[int, Sequence[int]],
    padding: Union[int, Sequence[int]],
    groups: int = 1,
    output_padding: Union[int, Sequence[int]] = 0,
) -> Tensor:
    sx, sy = normalize_stride(stride)
    px0, py0, px1, py1 = normalize_padding(padding)
    opx, opy = normalize_stride(output_padding)  # normalize output padding same as stride
    kx, ky = weight.shape[2:]
    gemm_x = conv2d_transpose_gemm_image(data, (kx, ky), (sx, sy), (px0, py0, px1, py1), groups, (opx, opy))
    gemm_w = conv2d_transpose_gemm_filter(weight, groups)
    gemm_y = matmul(gemm_x, gemm_w)

    p, q = data.shape[2:]
    h = (p - 1) * sx + -px0 - px1 + kx + output_padding[0]
    w = (q - 1) * sy + -py0 - py1 + ky + output_padding[1]
    y = conv2d_transpose_gemm_inverse(gemm_y, h, w)
    return y
