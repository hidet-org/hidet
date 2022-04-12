from typing import Union, Sequence
from ..common import Task, Operator, Tensor, DataLayout, Grid, tensor_input, compute, reduce, inline_compute, tensor_type, input_like


def conv2d_task(batch_size, in_channels, height, width, out_channels, kernel, padding, stride, input_layout=None, weight_layout=None, output_layout=None):
    input = tensor_input('input', 'float32', [batch_size, in_channels, height, width], scope='global', layout=input_layout)
    weight = tensor_input('weight', 'float32', [out_channels, in_channels, kernel[0], kernel[1]], scope='global', layout=weight_layout)
    padded = compute(
        name='pad',
        shape=[batch_size, in_channels, height + padding[0] + padding[2], weight + padding[1] + padding[3]],
        fcompute=lambda n, c, h, w: input.protect_read(indices=[n, c, h - padding[0], w - padding[1]], default_value=0.0))
    out_height = (height + padding[0] + padding[2] - kernel[0]) // stride[0] + 1
    out_width = (width + padding[1] + padding[3] - kernel[1]) // stride[1] + 1
    output = compute(
        name='out',
        shape=[batch_size, out_channels, out_height, out_width],
        fcompute=lambda n, c, h, w: reduce(
            shape=[in_channels, kernel[0], kernel[1]],
            fcompute=lambda rc, xx, yy: padded[n, rc, h * stride[0] + xx, w * stride[1] + yy] * weight.protect_read(indices=[c, rc, xx, yy], default_value=0.0),
            reduce_type='sum'
        ),
        scope='global',
        layout=output_layout
    )
    output = inline_compute(output)
    return Task(
        name='conv2d',
        computation=output,
        params=[input, weight, output],
        worker=Grid()
    )


class Conv2dOp(Operator):
    def __init__(
            self,
            input: Tensor,
            weight: Tensor,
            padding: Union[int, Sequence[int]],
            stride: Union[int, Sequence[int]]
    ):
        if input.shape[1] != weight.shape[1]:
            raise ValueError('Conv2d input shape {} and weight shape {} does not match.'.format(input.shape, weight.shape))
        batch_size, in_channels, height, width = input.shape
        out_channels = weight.shape[0]
        kernel = weight.shape[2:]
        if isinstance(padding, int):
            padding = [padding, padding, padding, padding]
        elif isinstance(padding, (tuple, list)) and len(padding) in [1, 2, 4]:
            if len(padding) == 1:
                padding = padding * 4
            elif len(padding) == 2:
                padding = [padding[0], padding[1], padding[0], padding[1]]
        else:
            raise ValueError('Expect int or (int, int) or (int, int, int, int) as padding, but got {}'.format(padding))
        if isinstance(stride, int):
            stride = [stride, stride]
        elif isinstance(stride, (tuple, list)) and len(stride) in [1, 2]:
            if len(stride) == 1:
                stride = stride * 2
        else:
            raise ValueError('Expect "stride" as int, (int, int), but got {}'.format(stride))
        task = conv2d_task(batch_size, in_channels, height, width, out_channels, kernel, padding, stride, input.layout, weight.layout)
        super().__init__([input, weight], task, padding=padding, stride=stride)


def conv2d(input: Tensor, weight, padding, stride) -> Tensor:
    return Conv2dOp(input, weight, padding, stride).get_output(0)
