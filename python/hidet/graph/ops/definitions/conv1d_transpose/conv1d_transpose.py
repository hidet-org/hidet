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
from typing import Optional
from hidet.ir.expr import if_then_else, logical_and
from hidet.ir.compute import compute, reduce
from hidet.graph.ops.definitions.utils import Task, Operator, Tensor, TensorNode
from hidet.graph.ops.definitions.utils import input_like, normalize_stride, normalize_padding, normalize_kernel


class Conv1dTransposeTask(Task):
    def __init__(
        self,
        data: TensorNode,
        weight: TensorNode,
        stride: Optional[int],
        padding: Optional[int],
        groups: Optional[int],
        output_padding: Optional[int],
    ):
        num_channels, out_channels, length_in = data.const_shape()
        out_channels, wc, kernel_size = weight.const_shape()
        s = normalize_stride(stride, dim=1)[0]
        p = normalize_padding(padding, dim=1)[0]
        k = normalize_kernel(kernel_size, dim=1)[0]
        op = normalize_padding(output_padding, dim=1)[0]
        channels_in = wc * groups
        l = (length_in - 1) * s - 2 * p + k + op

        if op >= s:
            raise ValueError(
                'Convd1dTranspose expects: output_padding < stride, \n'
                'but got output_padding, stride: {}, {}'.format(output_padding, s)
            )

        # output channels per group
        og = out_channels // groups

        # output
        output = compute(
            name='out',
            shape=[num_channels, channels_in, l],
            fcompute=lambda ni, ci, li: reduce(
                shape=[og, kernel_size],
                fcompute=lambda ogi, ki: if_then_else(
                    cond=logical_and(li + p >= ki, li + p < length_in * s + ki, (li + p - ki) % s == 0),
                    then_expr=(
                        data[ni, (ci // wc) * og + ogi, (li + p - ki) // s] * weight[(ci // wc) * og + ogi, ci % wc, ki]
                    ),
                    else_expr=data.type.dtype(0.0),
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
    ):
        super().__init__(
            inputs=[x, w],
            task=Conv1dTransposeTask(input_like(x, 'x'), input_like(w, 'w'), stride, padding, groups, output_padding),
            attributes={'stride': stride, 'padding': padding, 'groups': groups, 'output_padding': output_padding},
        )


def conv1d_transpose(
    data: Tensor,
    weight: Tensor,
    stride: Optional[int] = 1,
    padding: Optional[int] = 0,
    groups: Optional[int] = 1,
    output_padding: Optional[int] = 0,
) -> Tensor:
    s, p, op = stride, padding, output_padding
    return Conv1dTransposeOp(data, weight, s, p, groups, op).get_output(0)
