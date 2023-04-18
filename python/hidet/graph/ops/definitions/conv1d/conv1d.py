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
from typing import List, Union, Sequence
from hidet.graph.ops.definitions.utils import Task, Operator, Tensor, TensorNode
from hidet.graph.ops.definitions.utils import compute, input_like, normalize_stride, normalize_dilations, reduce
import pdb


class Conv1dTask(Task):
    def __init__(self, data: TensorNode, weight: TensorNode, stride: List[int], dilations: List[int], groups: int):
        # pdb.set_trace()
        n, c, l = data.const_shape()
        oc, wc, k = weight.const_shape()
        s = stride
        p = padding
        dil = dilations
        len_in = (l + 2 * p - dil * (k - 1) - 1) // s + 1
        if c % groups != 0 or oc % groups != 0:
            raise ValueError(
                'Conv1d expects: in_channels % groups == 0 and out_channels % groups == 0, \n'
                f'but got in_channels, out_channels, groups: {c}, {oc}, {groups}'
            )
        if wc * groups != c:
            raise ValueError(
                'Conv1d expect the weight has shape [out_channels, in_channels / groups, kx, ky], \n'
                f'got weight shape {[oc, wc, k]}, in_channels {c} and groups {groups}'
            )
        out_group_size = oc // groups
        output = compute(
            name='out',
            shape=[n, oc, len_in],
            fcompute=lambda ni, oci, li: reduce(
                shape=[wc, k],
                fcompute=lambda wci, ki: (
                    data[ni, (oci // out_group_size) * wc + wci, li * s + ki * dil] * weight[oci, wci, ki]
                ),
                reduce_type='sum',
            ),
        )
        self.channels = c
        self.stride = stride
        self.groups = groups
        super().__init__(name='conv1d', inputs=[data, weight], outputs=[output])


class Conv1dOp(Operator):
    def __init__(self, x: Tensor, w: Tensor, stride: Sequence[int], dilations: Union[int, Sequence[int]], groups: int):
        stride = normalize_stride(stride, dim=1)
        dilations = normalize_dilations(dilations, dim=1)
        super().__init__(
            inputs=[x, w],
            task=Conv1dTask(input_like(x, 'x'), input_like(w, 'w'), padding, stride, dilations, groups),
            attributes={'padding': padding, 'stride': stride, 'groups': groups, 'dilations': dilations},
        )


def conv1d(
    data: Tensor,
    weight: Tensor,
    stride: Union[int, Sequence[int]] = (1),
    dilations: Union[int, Sequence[int]] = (1),
    groups: int = 1,
) -> Tensor:
    return Conv1dOp(data, weight, padding, stride, dilations, groups).get_output(0)
