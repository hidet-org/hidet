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
from hidet import ir
from hidet.graph.ops.utils import Task, Operator, Tensor, TensorNode
from hidet.graph.ops.utils import compute, input_like, normalize_stride, normalize_dilations, reduce


# pylint: disable=too-many-locals
class Conv2dTask(Task):
    def __init__(
        self,
        data: TensorNode,
        weight: TensorNode,
        padding: List[int],
        stride: List[int],
        dilations: List[int],
        groups: int,
    ):
        from hidet.ir.compute.cops import pad

        # we assume that only data needs to have dynamic shape
        n, c, _, _ = data.shape
        oc, wc, kx, ky = weight.shape
        sx, sy = stride
        dilx, dily = dilations
        pad_h, pad_w = padding

        self._assert(
            ir.logical_or(c % groups == 0, oc % groups == 0),
            msg=(
                'Conv2d expect the in_channels % groups == 0 and out_channels % groups == 0, \n'
                'but got in_channels, out_channels, groups: {}, {}, {}'.format(c, oc, groups)
            ),
        )
        self._assert(
            wc * groups == c,
            msg=(
                'Conv2d expect the weight has shape [out_channels, in_channels / groups, kx, ky], \n'
                'got weight shape {}, in_channels {} and groups {}'.format([oc, wc, kx, ky], c, groups)
            ),
        )
        out_group_size = oc // groups

        pads = [0, 0, pad_h, pad_w, 0, 0, pad_h, pad_w]
        padded = pad(data, pads, value=0.0)  # only zero padding is needed right now

        _, _, ph, pw = padded.shape
        p, q = (ph - dilx * (kx - 1) - 1) // sx + 1, (pw - dily * (ky - 1) - 1) // sy + 1

        output = compute(
            name='out',
            shape=[n, oc, p, q],
            fcompute=lambda ni, oci, pi, qi: reduce(
                shape=[wc, kx, ky],
                fcompute=lambda wci, kxi, kyi: padded[
                    ni, (oci // out_group_size) * wc + wci, pi * sx + kxi * dilx, qi * sy + kyi * dily
                ]
                * weight[oci, wci, kxi, kyi],
                reduce_type='sum',
            ),
        )
        self.channels = c
        self.stride = stride
        self.groups = groups
        super().__init__(name='conv2d', inputs=[data, weight], outputs=[output])


class Conv2dChannelLastTask(Task):
    def __init__(self, data: TensorNode, weight: TensorNode, stride: List[int], dilations: List[int], groups: int):
        # pylint: disable=too-many-locals
        # we assume that only data needs to have dynamic shape
        n, h, w, c = data.shape
        oc, wc, kx, ky = weight.shape
        sx, sy = stride
        dilx, dily = dilations
        p, q = (h - dilx * (kx - 1) - 1) // sx + 1, (w - dily * (ky - 1) - 1) // sy + 1
        self._assert(
            ir.logical_or(c % groups == 0, oc % groups == 0),
            msg=(
                'Conv2d expect the in_channels % groups == 0 and out_channels % groups == 0, \n'
                'but got in_channels, out_channels, groups: {}, {}, {}'.format(c, oc, groups)
            ),
        )
        self._assert(
            wc * groups == c,
            msg=(
                'Conv2d expect the weight has shape [out_channels, in_channels / groups, kx, ky], \n'
                'got weight shape {}, in_channels {} and groups {}'.format([oc, wc, kx, ky], c, groups)
            ),
        )
        out_group_size = oc // groups
        output = compute(
            name='out',
            shape=[n, p, q, oc],
            fcompute=lambda ni, pi, qi, oci: reduce(
                shape=[wc, kx, ky],
                fcompute=lambda wci, kxi, kyi: (
                    data[ni, pi * sx + kxi * dilx, qi * sy + kyi * dily, (oci // out_group_size) * wc + wci]
                    * weight[oci, wci, kxi, kyi]
                ),
                reduce_type='sum',
            ),
        )
        self.channels = c
        self.stride = stride
        self.groups = groups
        super().__init__(name='conv2d_channel_last', inputs=[data, weight], outputs=[output])


class Conv2dOp(Operator):
    def __init__(
        self,
        x: Tensor,
        w: Tensor,
        padding: Sequence[int],
        stride: Sequence[int],
        dilations: Union[int, Sequence[int]],
        groups: int,
    ):
        stride = normalize_stride(stride)
        dilations = normalize_dilations(dilations)
        super().__init__(
            inputs=[x, w],
            attributes={'padding': padding, 'stride': stride, 'groups': groups, 'dilations': dilations},
            task=Conv2dTask(input_like(x, 'x'), input_like(w, 'w'), padding, stride, dilations, groups),
        )


class Conv2dChannelLastOp(Operator):
    def __init__(self, x: Tensor, w: Tensor, stride: Sequence[int], dilations: Union[int, Sequence[int]], groups: int):
        stride = normalize_stride(stride)
        dilations = normalize_dilations(dilations)
        super().__init__(
            inputs=[x, w],
            attributes={'stride': stride, 'groups': groups, 'dilations': dilations},
            task=Conv2dChannelLastTask(input_like(x, 'x'), input_like(w, 'w'), stride, dilations, groups),
        )


def conv2d(
    data: Tensor,
    weight: Tensor,
    stride: Sequence[int] = (1, 1),
    dilations: Sequence[int] = (1, 1),
    groups: int = 1,
    padding: Sequence[int] = (0, 0),
) -> Tensor:
    return Conv2dOp(data, weight, padding, stride, dilations, groups).get_output(0)


def conv2d_channel_last(
    data: Tensor,
    weight: Tensor,
    stride: Union[int, Sequence[int]] = (1, 1),
    dilations: Union[int, Sequence[int]] = (1, 1),
    groups: int = 1,
) -> Tensor:
    return Conv2dChannelLastOp(data, weight, stride, dilations, groups).get_output(0)
