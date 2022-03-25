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


def binary_elementwise(name, shape: Sequence[int], op: Callable[[Any, Any], Any]):
    x = tensor_input('x', 'float32', shape=shape)
    y = tensor_input('y', 'float32', shape=shape)
    z = compute(
        name='z',
        shape=shape,
        fcompute=lambda *indices: op(x[indices], y[indices])
    )
    return Task(
        name=name,
        computation=z,
        params=[x, y, z],
        params_type=[
            tensor_type('global', 'float32', x.shape, layout=DataLayout.row_major(x.shape)),
            tensor_type('global', 'float32', y.shape, layout=DataLayout.row_major(y.shape)),
            tensor_type('global', 'float32', z.shape, layout=DataLayout.row_major(z.shape)),
        ],
        worker=Grid()
    )

