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
from typing import Sequence, Union, Tuple, Optional
from hidet.ir.expr import if_then_else, LogicalAnd
from hidet.ir.compute import compute, reduce
from hidet.graph.ops.definitions.utils import Task, Operator, Tensor, TensorNode
from hidet.graph.ops.definitions.utils import input_like, normalize_stride, normalize_padding


class Conv2dTransposeTask(Task):
    def __init__(
        self,
        data: TensorNode,
        weight: TensorNode,
        stride: Optional[Tuple[int, int]],
        padding: Optional[Tuple[int, int]],
        groups: int,
        output_padding: Optional[Tuple[int, int]],
    ):
        n, oc, p, q = data.const_shape()
        oc, wc, kx, ky = weight.const_shape()
        g = groups
        c = wc * g
        sx, sy = stride
        px, py = padding
        opx, opy = output_padding
        h = (p - 1) * sx - 2 * px + kx + opx
        w = (q - 1) * sy - 2 * py + ky + opy

        if opx >= sx or opy >= sy:
            raise ValueError(
                'Conv2dTranspose expect the output_padding < stride, \n'
                'but got output_padding, stride: {}, {}'.format((opx, opy), (sx, sy))
            )
        if any(p < 0 for p in padding):
            raise ValueError('Negative padding is not supported.')

        og = oc // g  # output channels in each group

        output = compute(
            name='out',
            shape=[n, c, h, w],
            fcompute=lambda ni, ci, hi, wi: reduce(
                shape=[og, kx, ky],
                fcompute=lambda ogi, kxi, kyi: if_then_else(
                    cond=LogicalAnd.join(
                        hi + px >= kxi,
                        hi + px < p * sx + kxi,
                        (hi + px - kxi) % sx == 0,
                        wi + py >= kyi,
                        wi + py < q * sy + kyi,
                        (wi + py - kyi) % sy == 0,
                    ),
                    then_expr=(
                        data[ni, (ci // wc) * og + ogi, (hi + px - kxi) // sx, (wi + py - kyi) // sy]
                        * weight[(ci // wc) * og + ogi, ci % wc, kxi, kyi]
                    ),
                    else_expr=0.0,
                ),
                reduce_type='sum',
            ),
        )
        super().__init__(name='conv2d_transpose', inputs=[data, weight], outputs=[output])


class Conv2dTransposeOp(Operator):
    def __init__(
        self,
        x: Tensor,
        w: Tensor,
        stride: Optional[Tuple[int, int]],
        padding: Optional[Tuple[int, int]],
        groups: int,
        output_padding: Optional[Tuple[int, int]],
    ):
        super().__init__(
            inputs=[x, w],
            task=Conv2dTransposeTask(
                input_like(x, 'x'), input_like(w, 'w'), stride, padding, groups, output_padding, dilation
            ),
            attributes={
                'stride': stride,
                'padding': padding,
                'groups': groups,
                'output_padding': output_padding,
                'dilation': dilation,
            },
        )


def conv2d_transpose(
    data: Tensor,
    weight: Tensor,
    stride: Optional[Tuple[int, int]] = (1, 1),
    padding: Optional[Tuple[int, int]] = (0, 0),
    groups: Optional[int] = 1,
    output_padding: Optional[Tuple[int, int]] = (0, 0),
) -> Tensor:
    sx, sy = stride
    px, py = padding
    opx, opy = output_padding
    return Conv2dTransposeOp(data, weight, (sx, sy), (px, py), groups, (opx, opy)).get_output(0)
