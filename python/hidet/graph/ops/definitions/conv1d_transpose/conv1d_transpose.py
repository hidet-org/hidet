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
        stride: Tuple[int],
        padding: Tuple[int],
        groups: int,
        output_padding: Tuple[int],
    ):
        num_channels, out_channels, length_in = data.const_shape()
        out_channels, wc, kernel_size = weight.const_shape()
        channels_in = wc * groups
        sx = stride[0]
        px = padding[0]
        l = (length_in - 1) * sx - px - px + kernel_size + output_padding[0]

        if output_padding >= stride or output_padding[0] >= stride[0]:
            raise ValueError(
                'Convd1dTranspose expects: output_padding < stride, \n'
                f'but got output_padding, stride: {output_padding}, {stride}'
            )
        if any(p < 0 for p in padding):
            raise ValueError('Negative padding is not supported.')

        # output channels per group
        og = out_channels // groups

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
        self, x: Tensor, w: Tensor, stride: Tuple[int], padding: Tuple[int], groups: int, output_padding: Tuple[int]
    ):
        super().__init__(
            inputs=[x, w],
            task=Conv1dTransposeTask(input_like(x, 'x'), input_like(w, 'w'), stride, padding, groups, output_padding),
            attributes={'stride': stride, 'padding': padding, 'groups': groups, 'output_padding': output_padding},
        )


def conv1d_transpose(
    data: Tensor,
    weight: Tensor,
    stride: Optional[Union[int, Sequence[int]]] = 1,
    padding: Optional[Union[int, Sequence[int]]] = 0,
    groups: Optional[int] = 1,
    output_padding: Optional[Union[int, Sequence[int]]] = 0,
) -> Tensor:
    sx = normalize_stride(stride)
    px = normalize_padding(padding)
    opx = normalize_stride(output_padding)  # normalize output padding same as stride
    return Conv1dTransposeOp(data, weight, (sx), (px), groups, (opx)).get_output(0)
