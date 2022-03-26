from typing import List, Tuple, Union, Sequence
from hidet.ir.dialects.compute import tensor_input, compute, reduce
from hidet.ir.layout import DataLayout
from hidet.ir.task import Task, Grid
from hidet.ir.type import tensor_type
from hidet.ir.functors import inline_compute
from hidet.utils.info import float_type_min_value


def normalize(v, rank=2):
    if not isinstance(v, (tuple, list)):
        v = [v for _ in range(rank)]
    assert len(v) == rank
    return v


def pool2d(shape: Sequence[int],
           kernel: Union[Sequence[int], int],
           strides: Union[Sequence[int], int],
           padding: Union[Sequence[int], int],
           reduce_type: str):
    assert reduce_type in ['max', 'avg']
    kernel = normalize(kernel)
    strides = normalize(strides)
    padding = normalize(padding)
    batch_size, channels, height, width = shape
    out_height = (height + 2 * padding[0] - kernel[0]) // strides[0] + 1
    out_width = (width + 2 * padding[1] - kernel[1]) // strides[1] + 1
    x = tensor_input('x', 'float32', shape=[batch_size, channels, height, width])
    pad = compute(
        name='pad',
        shape=[batch_size, channels, height + 2 * padding[0], width + 2 * padding[1]],
        fcompute=lambda n, c, h, w: x.protect_read(indices=[n, c, h - padding[0], w - padding[1]], default_value=float_type_min_value())
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
    return Task(
        name='{}_pool2d'.format(reduce_type),
        computation=y,
        params=[x, y],
        params_type=[
            tensor_type('global', 'float32', x.shape, layout=DataLayout.row_major(x.shape)),
            tensor_type('global', 'float32', y.shape, layout=DataLayout.row_major(y.shape)),
        ],
        worker=Grid()
    )


def max_pool2d(
        shape: Sequence[int],
        kernel: Union[Sequence[int], int],
        strides: Union[Sequence[int], int],
        padding: Union[Sequence[int], int]):
    kernel = normalize(kernel)
    strides = normalize(strides)
    padding = normalize(padding)
    batch_size, channels, height, width = shape
    out_height = (height + 2 * padding[0] - kernel[0]) // strides[0] + 1
    out_width = (width + 2 * padding[1] - kernel[1]) // strides[1] + 1
    x = tensor_input('x', 'float32', shape=[batch_size, channels, height, width])
    pad = compute(
        name='pad',
        shape=[batch_size, channels, height + 2 * padding[0], width + 2 * padding[1]],
        fcompute=lambda n, c, h, w: x.protect_read(indices=[n, c, h - padding[0], w - padding[1]], default_value=float_type_min_value())
    )
    y = compute(
        name='y',
        shape=[batch_size, channels, out_height, out_width],
        fcompute=lambda n, c, h, w: reduce(
            shape=[kernel[0], kernel[1]],
            fcompute=lambda rx, ry: pad[n, c, h * strides[0] + rx, w * strides[1] + ry],
            reduce_type='max'
        )
    )
    y = inline_compute(y)
    return Task(
        name='max_pool2d',
        computation=y,
        params=[x, y],
        params_type=[
            tensor_type('global', 'float32', x.shape, layout=DataLayout.row_major(x.shape)),
            tensor_type('global', 'float32', y.shape, layout=DataLayout.row_major(y.shape)),
        ],
        worker=Grid()
    )


def avg_pool2d(
        shape: Sequence[int],
        kernel: Union[Sequence[int], int],
        strides: Union[Sequence[int], int],
        padding: Union[Sequence[int], int]):
    kernel = normalize(kernel)
    strides = normalize(strides)
    padding = normalize(padding)
    batch_size, channels, height, width = shape
    out_height = (height + 2 * padding[0] - kernel[0]) // strides[0] + 1
    out_width = (width + 2 * padding[1] - kernel[1]) // strides[1] + 1
    x = tensor_input('x', 'float32', shape=[batch_size, channels, height, width])
    pad = compute(
        name='pad',
        shape=[batch_size, channels, height + 2 * padding[0], width + 2 * padding[1]],
        fcompute=lambda n, c, h, w: x.protect_read(indices=[n, c, h - padding[0], w - padding[1]], default_value=float_type_min_value())
    )
    y = compute(
        name='y',
        shape=[batch_size, channels, out_height, out_width],
        fcompute=lambda n, c, h, w: reduce(
            shape=[kernel[0], kernel[1]],
            fcompute=lambda rx, ry: pad[n, c, h * strides[0] + rx, w * strides[1] + ry],
            reduce_type='avg'
        )
    )
    y = inline_compute(y)
    return Task(
        name='avg_pool2d',
        computation=y,
        params=[x, y],
        params_type=[
            tensor_type('global', 'float32', x.shape, layout=DataLayout.row_major(x.shape)),
            tensor_type('global', 'float32', y.shape, layout=DataLayout.row_major(y.shape)),
        ],
        worker=Grid()
    )
