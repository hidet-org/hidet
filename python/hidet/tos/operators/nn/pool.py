from typing import Union, Sequence

from ..common import Task, Operator, Tensor, TensorInput, Grid, compute, reduce, inline_compute, input_like, normalize_stride, normalize_kernel, normalize_padding


def pool2d_task(x: TensorInput, kernel, strides, padding, reduce_type: str) -> Task:
    assert reduce_type in ['max', 'avg']
    kernel = normalize_kernel(kernel)
    strides = normalize_stride(strides)
    padding = normalize_padding(padding)
    batch_size, channels, height, width = x.const_shape()
    out_height = (height + padding[0] + padding[2] - kernel[0]) // strides[0] + 1
    out_width = (width + padding[1] + padding[3] - kernel[1]) // strides[1] + 1
    pad_value = 0.0 if reduce_type == 'avg' else -1e30
    pad = compute(
        name='pad',
        shape=[batch_size, channels, height + 2 * padding[0], width + 2 * padding[1]],
        fcompute=lambda n, c, h, w: x.protect_read(indices=[n, c, h - padding[0], w - padding[1]], default_value=pad_value)
    )
    y = compute(
        name='y',
        shape=[batch_size, channels, out_height, out_width],
        fcompute=lambda n, c, h, w: reduce(
            shape=[kernel[0], kernel[1]],
            fcompute=lambda rx, ry: pad[n, c, h * strides[0] + rx, w * strides[1] + ry],
            reduce_type=reduce_type
        ),
        scope='global'
    )
    y = inline_compute(y)
    return Task(
        name='{}_pool2d'.format(reduce_type),
        computation=y,
        params=[x, y],
        worker=Grid()
    )


class MaxPool2dOp(Operator):
    def __init__(self,
                 input: Tensor,
                 kernel: Union[int, Sequence[int]],
                 stride: Union[int, Sequence[int]],
                 padding: Union[int, Sequence[int]]
                 ):
        super().__init__(
            inputs=[input],
            task=pool2d_task(input_like(input, 'x'), kernel, stride, padding, reduce_type='max'),
            kernel=kernel,
            stride=stride,
            padding=padding
        )


class AvgPool2dOp(Operator):
    def __init__(self,
                 input: Tensor,
                 kernel: Union[int, Sequence[int]],
                 stride: Union[int, Sequence[int]],
                 padding: Union[int, Sequence[int]]
                 ):
        super().__init__(
            inputs=[input],
            task=pool2d_task(input_like(input, 'x'), kernel, stride, padding, reduce_type='avg'),
            kernel=kernel,
            stride=stride,
            padding=padding
        )


def max_pool2d(input: Tensor, kernel, stride, padding) -> Tensor:
    return MaxPool2dOp(input, kernel, stride, padding).get_output(0)


def avg_pool2d(input: Tensor, kernel, stride, padding) -> Tensor:
    return AvgPool2dOp(input, kernel, stride, padding).get_output(0)
