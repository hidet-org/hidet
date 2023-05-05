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
from hidet.ir.expr import if_then_else, logical_and
from hidet.ir.compute import compute, reduce
from hidet.graph.ops.definitions.utils import Task, Operator, Tensor, TensorNode
from hidet.graph.ops.definitions.utils import input_like, normalize_stride, normalize_padding


class Conv3dTransposeTask(Task):
    def __init__(
        self,
        data: TensorNode,
        weight: TensorNode,
        stride: Tuple[int, int, int],
        padding: Tuple[int, int, int, int, int, int],
        groups: int,
        output_padding: Tuple[int, int, int],
    ):
        n, oc, r, p, q = data.const_shape()
        oc, wc, kz, kx, ky = weight.const_shape()
        c = wc * groups
        sz, sx, sy = stride
        pz0, px0, py0, pz1, px1, py1 = padding
        z = (r - 1) * sz - pz0 - pz1 + kz + output_padding[0]
        h = (p - 1) * sx - px0 - px1 + kx + output_padding[1]
        w = (q - 1) * sy - py0 - py1 + ky + output_padding[2]

        if output_padding[0] >= stride[0] or output_padding[1] >= stride[1] or output_padding[2] >= stride[2]:
            raise ValueError(
                'Conv3dTranspose expect the output_padding < stride, \n'
                'but got output_padding, stride: {}, {}'.format(output_padding, stride)
            )
        if any(p < 0 for p in padding):
            raise ValueError('Negative padding is not supported.')

        og = oc // groups  # output channels in each group
        output = compute(
            name='out',
            shape=[n, c, z, h, w],
            fcompute=lambda ni, ci, zi, hi, wi: reduce(
                shape=[og, kz, kx, ky],
                fcompute=lambda ogi, kzi, kxi, kyi: if_then_else(
                    cond=logical_and(
                        zi + pz0 >= kzi,
                        zi + pz0 < r * sz + kzi,
                        (zi + pz0 - kzi) % sz == 0,
                        hi + px0 >= kxi,
                        hi + px0 < p * sx + kxi,
                        (hi + px0 - kxi) % sx == 0,
                        wi + py0 >= kyi,
                        wi + py0 < q * sy + kyi,
                        (wi + py0 - kyi) % sy == 0,
                    ),
                    then_expr=(
                        data[
                            ni,
                            (ci // wc) * og + ogi,
                            (zi + pz0 - kzi) // sz,
                            (hi + px0 - kxi) // sx,
                            (wi + py0 - kyi) // sy,
                        ]
                        * weight[(ci // wc) * og + ogi, ci % wc, kzi, kxi, kyi]
                    ),
                    else_expr=0.0,
                ),
                reduce_type='sum',
            ),
        )
        super().__init__(name='conv3d_transpose', inputs=[data, weight], outputs=[output])


class Conv3dTransposeOp(Operator):
    def __init__(
        self,
        x: Tensor,
        w: Tensor,
        stride: Tuple[int, int, int],
        padding: Tuple[int, int, int, int, int, int],
        groups: int,
        output_padding: Tuple[int, int, int],
    ):
        super().__init__(
            inputs=[x, w],
            task=Conv3dTransposeTask(input_like(x, 'x'), input_like(w, 'w'), stride, padding, groups, output_padding),
            attributes={'stride': stride, 'padding': padding, 'groups': groups, 'output_padding': output_padding},
        )


def conv3d_transpose(
    data: Tensor,
    weight: Tensor,
    stride: Union[int, Sequence[int]] = (1, 1, 1),
    padding: Union[int, Sequence[int]] = (0, 0, 0),
    groups: int = 1,
    output_padding: Union[int, Sequence[int]] = 0,
) -> Tensor:
    sz, sx, sy = normalize_stride(stride, dim=3)
    pz0, px0, py0, pz1, px1, py1 = normalize_padding(padding, dim=3)
    opz, opx, opy = normalize_stride(output_padding, dim=3)
    return Conv3dTransposeOp(
        data, weight, (sz, sx, sy), (pz0, px0, py0, pz1, px1, py1), groups, (opz, opx, opy)
    ).get_output(0)
