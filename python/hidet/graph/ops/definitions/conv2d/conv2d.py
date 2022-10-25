from typing import List, Union
from hidet.graph.ops.definitions.utils import Task, Operator, Tensor, TensorNode
from hidet.graph.ops.definitions.utils import compute, input_like, normalize_stride, reduce


class Conv2dTask(Task):
    def __init__(self, data: TensorNode, weight: TensorNode, stride: List[int], groups: int):
        # pylint: disable=too-many-locals
        n, c, h, w = data.const_shape()
        oc, wc, kx, ky = weight.const_shape()
        sx, sy = stride
        p, q = (h - kx) // sx + 1, (w - ky) // sy + 1
        if c % groups != 0 or oc % groups != 0:
            raise ValueError('Conv2d expect the in_channels % groups == 0 and out_channels % groups == 0, \n'
                             'but got in_channels, out_channels, groups: {}, {}, {}'.format(c, oc, groups))
        if wc * groups != c:
            raise ValueError('Conv2d expect the weight has shape [out_channels, in_channels / groups, kx, ky], \n'
                             'got weight shape {}, in_channels {} and groups {}'.format([oc, wc, kx, ky], c, groups))
        out_group_size = oc // groups
        output = compute(
            name='out',
            shape=[n, oc, p, q],
            fcompute=lambda ni, oci, pi, qi: reduce(
                shape=[wc, kx, ky],
                fcompute=lambda wci, kxi, kyi: (
                  data[ni, (oci // out_group_size) * wc + wci, pi * sx + kxi, qi * sy + kyi]
                  *
                  weight[oci, wci, kxi, kyi]
                ),
                reduce_type='sum'
            )
        )
        self.channels = c
        self.stride = stride
        self.groups = groups
        super().__init__(
            name='conv2d',
            inputs=[data, weight],
            outputs=[output],
        )


class Conv2dOp(Operator):
    def __init__(self, x: Tensor, w: Tensor, stride: List[int], groups: int):
        stride = normalize_stride(stride)
        super().__init__(
            inputs=[x, w],
            task=Conv2dTask(input_like(x, 'x'), input_like(w, 'w'), stride, groups),
            attributes={
                'stride': stride,
                'groups': groups
            }
        )


def conv2d(data: Tensor, weight: Tensor, stride: Union[int, List[int]], groups: int = 1) -> Tensor:
    return Conv2dOp(data, weight, stride, groups).get_output(0)
