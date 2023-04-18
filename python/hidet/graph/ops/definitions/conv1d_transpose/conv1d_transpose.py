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


class Conv1dTransposeTask(Task):
    def __init__(
        self,
        data: TensorNode,
        weight: TensorNode,
        stride: Optional[int],
        padding: Optional[int],
        groups: Optional[int],
        output_padding: Optional[int],
        dilation: Optional[int],
    ):
        num_channels, out_channels, length_in = data.const_shape()
        out_channels, wc, kernel_size = weight.const_shape()
        if (type(padding) and type(stride) and type(dilation) and type(groups)) is not int:
            px, sx, dil, g = padding[0], stride[0], dilation[0], groups[0]
        else:
            px, sx, dil, g = padding, stride, dilation, groups
        channels_in = wc * g
        l = (length_in - 1) * sx - 2 * px + dil * (kernel_size - 1) + output_padding + 1

        if output_padding >= sx:
            raise ValueError(
                'Convd1dTranspose expects: output_padding < stride, \n'
                f'but got output_padding, stride: {output_padding}, {sx}'
            )

        # output channels per group
        og = out_channels // g

        # output
        output = compute(
            name='out',
            shape=[num_channels, channels_in, l],
            fcompute=lambda ni, ci, li: reduce(
                shape=[og, kernel_size],
                fcompute=lambda ogi, ki: if_then_else(
                    cond=LogicalAnd.join(li + px >= ki, li + px < length_in * sx + ki, (li + px - ki) % sx == 0),
                    then_expr=(
                        data[ni, (ci // wc) * og + ogi, (li + px - ki) // sx]
                        * weight[(ci // wc) * og + ogi, ci % wc, ki]
                    ),
                    else_expr=0.0,
                ),
                reduce_type='sum',
            ),
        )
        super().__init__(name='conv1d_transpose', inputs=[data, weight], outputs=[output])


class Conv1dTransposeOp(Operator):
    def __init__(
        self,
        x: Tensor,
        w: Tensor,
        stride: Optional[int],
        padding: Optional[int],
        groups: Optional[int],
        output_padding: Optional[int],
        dilation: Optional[int],
    ):
        super().__init__(
            inputs=[x, w],
            task=Conv1dTransposeTask(
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


def conv1d_transpose(
    data: Tensor,
    weight: Tensor,
    stride: Optional[int] = 1,
    padding: Optional[int] = 0,
    groups: Optional[int] = 1,
    output_padding: Optional[int] = 0,
    dilation: Optional[int] = 1,
) -> Tensor:
    sx = stride
    px = padding
    opx = output_padding
    dil = dilation
    return Conv1dTransposeOp(data, weight, sx, px, groups, opx, dil).get_output(0)
