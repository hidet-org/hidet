from typing import Union, Sequence
from hidet.ir.expr import convert

from .utils import Task, Operator, Tensor, TensorNode, compute, reduce, inline_compute, input_like, normalize_stride, normalize_kernel, normalize_padding


class Pool2dTask(Task):
    def __init__(self, x: TensorNode, kernel, strides, padding, reduce_type: str):
        assert reduce_type in ['max', 'avg']
        kernel = normalize_kernel(kernel)
        strides = normalize_stride(strides)
        padding = normalize_padding(padding)
        batch_size, channels, height, width = x.const_shape()
        out_height = (height + padding[0] + padding[2] - kernel[0]) // strides[0] + 1
        out_width = (width + padding[1] + padding[3] - kernel[1]) // strides[1] + 1
        pad_value = convert(0.0 if reduce_type == 'avg' else -1e30, dtype=x.data_type.scalar_type)
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
            )
        )
        y = inline_compute(y)
        super().__init__(
            name='{}_pool2d'.format(reduce_type),
            inputs=[x],
            outputs=[y]
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
            task=Pool2dTask(input_like(input, 'x'), kernel, stride, padding, reduce_type='max'),
            attributes={
                'kernel': kernel,
                'stride': stride,
                'padding': padding
            }
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
            task=Pool2dTask(input_like(input, 'x'), kernel, stride, padding, reduce_type='avg'),
            attributes={
                'kernel': kernel,
                'stride': stride,
                'padding': padding
            }
        )


def max_pool2d(input: Tensor, kernel, stride, padding) -> Tensor:
    return MaxPool2dOp(input, kernel, stride, padding).get_output(0)


def avg_pool2d(input: Tensor, kernel, stride, padding) -> Tensor:
    return AvgPool2dOp(input, kernel, stride, padding).get_output(0)
