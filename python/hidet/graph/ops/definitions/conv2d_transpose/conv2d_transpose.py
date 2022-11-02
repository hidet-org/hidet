from typing import Sequence, Union
from hidet.ir.expr import if_then_else, And
from hidet.graph.ops.definitions.utils import Task, Operator, Tensor, TensorNode
from hidet.graph.ops.definitions.utils import compute, input_like, normalize_stride, reduce, normalize_padding


class Conv2dTransposeTask(Task):
    def __init__(
        self,
        data: TensorNode,
        weight: TensorNode,
        stride: Sequence[int],  # [sx, sy]
        padding: Sequence[int],  # [px0, py0, px1, py1]
        groups: int,
        output_padding: Sequence[int],  # [opx, opy]
    ):
        n, oc, p, q = data.const_shape()
        oc, wc, kx, ky = weight.const_shape()
        c = wc * groups
        sx, sy = stride
        px0, py0, px1, py1 = padding
        h = (p - 1) * sx + -px0 - px1 + kx + output_padding[0]
        w = (q - 1) * sy + -py0 - py1 + ky + output_padding[1]

        if output_padding[0] >= stride[0] or output_padding[1] >= stride[1]:
            raise ValueError(
                'Conv2dTranspose expect the output_padding < stride, \n'
                'but got output_padding, stride: {}, {}'.format(output_padding, stride)
            )
        if any(p < 0 for p in padding):
            raise ValueError('Negative padding is not supported.')

        out_group_size = oc // groups
        output = compute(
            name='out',
            shape=[n, c, h, w],
            fcompute=lambda ni, ci, hi, wi: reduce(
                shape=[out_group_size, kx, ky],
                fcompute=lambda ogi, kxi, kyi: if_then_else(
                    cond=And.join(
                        hi + px0 >= kxi,
                        hi + px0 < p * sx + kxi,
                        (hi + px0 - kxi) % sx == 0,
                        wi + py0 >= kyi,
                        wi + py0 < q * sy + kyi,
                        (wi + py0 - kyi) % sy == 0,
                    ),
                    then_expr=(
                        data[ni, (ci // wc) * out_group_size + ogi, (hi + px0 - kxi) // sx, (wi + py0 - kyi) // sy]
                        * weight[(ci // wc) * out_group_size + ogi, ci % wc, kxi, kyi]
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
        stride: Sequence[int],
        padding: Sequence[int],
        groups: int,
        output_padding: Sequence[int],
    ):
        stride = normalize_stride(stride)
        padding = normalize_padding(padding)
        output_padding = normalize_stride(output_padding)  # normalize output padding same as stride
        super().__init__(
            inputs=[x, w],
            task=Conv2dTransposeTask(input_like(x, 'x'), input_like(w, 'w'), stride, padding, groups, output_padding),
            attributes={'stride': stride, 'padding': padding, 'groups': groups, 'output_padding': output_padding},
        )


def conv2d_transpose(
    data: Tensor,
    weight: Tensor,
    stride: Union[int, Sequence[int]],
    padding: Union[int, Sequence[int]],
    groups: int = 1,
    output_padding: Union[int, Sequence[int]] = 0,
) -> Tensor:
    return Conv2dTransposeOp(data, weight, stride, padding, groups, output_padding).get_output(0)
