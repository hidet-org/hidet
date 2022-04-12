from typing import List, Tuple, Union, Sequence, Callable, Any
from hidet.ir.dialects.compute import tensor_input, compute, reduce
from hidet.ir.layout import DataLayout
from hidet.ir.task import Task, Grid
from hidet.ir.type import tensor_type
from hidet.ir.functors import inline_compute
from hidet.utils.info import float_type_min_value


def unary_elementwise(name, shape: Sequence[int], op: Callable[[Any], Any]):
    x = tensor_input('x', 'float32', shape=shape, scope='global')
    y = compute(
        name='y',
        shape=shape,
        fcompute=lambda *indices: op(x.__getitem__(indices)),
        scope='global'
    )
    return Task(
        name=name,
        computation=y,
        params=[x, y],
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
    x = tensor_input('x', 'float32', shape=x_layout.shape, scope='global', layout=x_layout)
    y = tensor_input('y', 'float32', shape=y_layout.shape, scope='global', layout=y_layout)
    z_shape = broadcast_shape(x.data_type.shape, y.data_type.shape)
    if z_layout is None:
        z_layout = DataLayout.row_major(z_shape)
    else:
        assert [a == b for a, b in zip(z_layout.shape, z_shape)]

    def imap(indices, shape):
        # used to support broadcast
        pad_dim = len(z_shape) - len(shape)
        indices = list(indices[pad_dim:])
        for idx, dim in enumerate(shape):
            if int(dim) == 1:
                indices[idx] = 0
        return indices

    z = compute(
        name='z',
        shape=z_shape,
        fcompute=lambda *indices: op(x[imap(indices, x.data_type.shape)], y[imap(indices, y.data_type.shape)]),
        scope='global'
    )
    return Task(
        name=name,
        computation=z,
        params=[x, y, z],
        worker=Grid()
    )
