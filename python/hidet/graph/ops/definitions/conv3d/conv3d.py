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


class Conv3dTask(Task):
    def __init__(self, data: TensorNode, weight: TensorNode, stride: List[int], dilations: List[int], groups: int):
        # pylint: disable=too-many-locals
        n, c, d, h, w = data.const_shape()
        oc, wc, kz, kx, ky = weight.const_shape()
        sz, sx, sy = stride
        dilz, dilx, dily = dilations
        r, p, q = (
            (d - dilz * (kz - 1) - 1) // sz + 1,
            (h - dilx * (kx - 1) - 1) // sx + 1,
            (w - dily * (ky - 1) - 1) // sy + 1,
        )
        if c % groups != 0 or oc % groups != 0:
            raise ValueError(
                'Conv3d expect the in_channels % groups == 0 and out_channels % groups == 0, \n'
                'but got in_channels, out_channels, groups: {}, {}, {}'.format(c, oc, groups)
            )
        if wc * groups != c:
            raise ValueError(
                'Conv3d expect the weight has shape [out_channels, in_channels / groups, kx, ky], \n'
                'got weight shape {}, in_channels {} and groups {}'.format([oc, wc, kx, ky], c, groups)
            )
        out_group_size = oc // groups
        output = compute(
            name='out',
            shape=[n, oc, r, p, q],
            fcompute=lambda ni, oci, ri, pi, qi: reduce(
                shape=[wc, kz, kx, ky],
                fcompute=lambda wci, kzi, kxi, kyi: (
                    data[
                        ni,
                        (oci // out_group_size) * wc + wci,
                        ri * sz + kzi * dilz,
                        pi * sx + kxi * dilx,
                        qi * sy + kyi * dily,
                    ]
                    * weight[oci, wci, kzi, kxi, kyi]
                ),
                reduce_type='sum',
            ),
        )
        self.channels = c
        self.stride = stride
        self.groups = groups
        super().__init__(name='conv3d', inputs=[data, weight], outputs=[output])


class Conv3dOp(Operator):
    def __init__(self, x: Tensor, w: Tensor, stride: Sequence[int], dilations: Union[int, Sequence[int]], groups: int):
        stride = normalize_stride(stride, dim=3)
        if isinstance(dilations, int):
            dilations = [dilations, dilations, dilations]
        super().__init__(
            inputs=[x, w],
            task=Conv3dTask(input_like(x, 'x'), input_like(w, 'w'), stride, dilations, groups),
            attributes={'stride': stride, 'groups': groups, 'dilations': dilations},
        )


def conv3d(
    data: Tensor,
    weight: Tensor,
    stride: Union[int, Sequence[int]],
    dilations: Union[int, Sequence[int]],
    groups: int = 1,
) -> Tensor:
    return Conv3dOp(data, weight, stride, dilations, groups).get_output(0)
