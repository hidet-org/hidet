from typing import List, Tuple, Union, Sequence, Callable, Any
from hidet.ir.dialects.compute import tensor_input, compute, reduce
from hidet.ir.layout import DataLayout
from hidet.ir.task import Task, Grid
from hidet.ir.type import tensor_type
from hidet.ir.functors import inline_compute
from hidet.utils.info import float_type_min_value


def unary_elementwise(name, shape: Sequence[int], op: Callable[[Any], Any]):
    x = tensor_input('x', 'float32', shape=shape)
    y = compute(
        name='y',
        shape=shape,
        fcompute=lambda *indices: op(x.__getitem__(indices))
    )
    return Task(
        name=name,
        computation=y,
        params=[x, y],
        params_type=[
            tensor_type('global', 'float32', x.shape, layout=DataLayout.row_major(x.shape)),
            tensor_type('global', 'float32', y.shape, layout=DataLayout.row_major(y.shape)),
        ],
        worker=Grid()
    )


def broadcast_shape(x_shape, y_shape):
    x_shape = [int(v) for v in x_shape]
    y_shape = [int(v) for v in y_shape]
    orig_shapes = x_shape, y_shape
    while len(x_shape) < len(y_shape):
        x_shape = [1] + x_shape
    while len(y_shape) < len(x_shape):
        y_shape = [1] + y_shape
    result_shape = []
    for p, q in zip(x_shape, y_shape):
        if p != q and p != 1 and q != 1:
            raise ValueError('can not broadcast two arrays with shape {} and {}'.format(orig_shapes[0], orig_shapes[1]))
        result_shape.append(max(p, q))
    return result_shape


def binary_elementwise(name, x_layout, y_layout, op: Callable[[Any, Any], Any], z_layout=None):
    x = tensor_input('x', 'float32', shape=x_layout.shape)
    y = tensor_input('y', 'float32', shape=y_layout.shape)
    z_shape = broadcast_shape(x.shape, y.shape)
    if z_layout is None:
        z_layout = DataLayout.row_major(z_shape)
    else:
        assert [a == b for a, b in zip(z_layout.shape, z_shape)]

    def imap(indices):
        # used to support broadcast
        return [i if d != 1 else 0 for i, d in zip(indices, z_shape)]

    z = compute(
        name='z',
        shape=z_shape,
        fcompute=lambda *indices: op(x[imap(indices)], y[imap(indices)])
    )
    return Task(
        name=name,
        computation=z,
        params=[x, y, z],
        params_type=[
            tensor_type('global', 'float32', x.shape, layout=x_layout),
            tensor_type('global', 'float32', y.shape, layout=y_layout),
            tensor_type('global', 'float32', z.shape, layout=z_layout),
        ],
        worker=Grid()
    )
