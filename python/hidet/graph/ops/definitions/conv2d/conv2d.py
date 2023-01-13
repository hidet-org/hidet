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
from hidet.graph.ops.definitions.utils import compute, input_like, normalize_stride, reduce


class Conv2dTask(Task):
    def __init__(self, data: TensorNode, weight: TensorNode, stride: List[int], dilations: List[int], groups: int):
        # pylint: disable=too-many-locals
        n, c, h, w = data.const_shape()
        oc, wc, kx, ky = weight.const_shape()
        sx, sy = stride
        dilx, dily = dilations
        p, q = (h - dilx * (kx - 1) - 1) // sx + 1, (w - dily * (ky - 1) - 1) // sy + 1
        if c % groups != 0 or oc % groups != 0:
            raise ValueError(
                'Conv2d expect the in_channels % groups == 0 and out_channels % groups == 0, \n'
                'but got in_channels, out_channels, groups: {}, {}, {}'.format(c, oc, groups)
            )
        if wc * groups != c:
            raise ValueError(
                'Conv2d expect the weight has shape [out_channels, in_channels / groups, kx, ky], \n'
                'got weight shape {}, in_channels {} and groups {}'.format([oc, wc, kx, ky], c, groups)
            )
        out_group_size = oc // groups
        output = compute(
            name='out',
            shape=[n, oc, p, q],
            fcompute=lambda ni, oci, pi, qi: reduce(
                shape=[wc, kx, ky],
                fcompute=lambda wci, kxi, kyi: (
                    data[ni, (oci // out_group_size) * wc + wci, pi * sx + kxi * dilx, qi * sy + kyi * dily]
                    * weight[oci, wci, kxi, kyi]
                ),
                reduce_type='sum',
            ),
        )
        self.channels = c
        self.stride = stride
        self.groups = groups
        super().__init__(name='conv2d', inputs=[data, weight], outputs=[output])


class Conv2dOp(Operator):
    def __init__(self, x: Tensor, w: Tensor, stride: Sequence[int], dilations: Union[int, Sequence[int]], groups: int):
        stride = normalize_stride(stride)
        if isinstance(dilations, int):
            dilations = [dilations, dilations]
        super().__init__(
            inputs=[x, w],
            task=Conv2dTask(input_like(x, 'x'), input_like(w, 'w'), stride, dilations, groups),
            attributes={'stride': stride, 'groups': groups, 'dilations': dilations},
        )


def conv2d(
    data: Tensor,
    weight: Tensor,
    stride: Union[int, Sequence[int]],
    dilations: Union[int, Sequence[int]] = (1, 1),
    groups: int = 1,
) -> Tensor:
    return Conv2dOp(data, weight, stride, dilations, groups).get_output(0)
