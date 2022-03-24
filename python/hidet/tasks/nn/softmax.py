from typing import List
from hidet.ir.dialects.compute import tensor_input, compute, reduce
from hidet.ir.layout import DataLayout
from hidet.ir.task import Task, Grid
from hidet.ir.type import tensor_type
from hidet.ir.functors import inline_compute
from hidet.ir.primitives import expf


def softmax(shape: List[int], axis: int):
    axis = (axis + len(shape)) % len(shape)
    # todo: support other shape
    assert len(shape) == 4 and axis == 1
    m, n, p, q = shape
    x = tensor_input('x', 'float32', shape=[m, n, p, q])
    mx = compute(
        name='mx',
        shape=[m, p, q],
        fcompute=lambda i, r, s: reduce(
            shape=[n],
            fcompute=lambda j: x[i, j, r, s],
            reduce_type='max'
        )
    )
    e = compute(
        name='e',
        shape=[m, n, p, q],
        fcompute=lambda i, j, r, s: expf(x[i, j, r, s] - mx[i, r, s])
    )
    se = compute(
        name='se',
        shape=[m, p, q],
        fcompute=lambda i, r, s: reduce(
            shape=[n],
            fcompute=lambda j: e[i, j, r, s],
            reduce_type='sum'
        )
    )
    out = compute(
        name='out',
        shape=[m, n, p, q],
        fcompute=lambda i, j, r, s: e[i, j, r, s] / se[i, r, s]
    )
    return Task(
        name='softmax',
        computation=out,
        params=[x, out],
        params_type=[
            tensor_type(scope='global', dtype='float32', layout=DataLayout.row_major([m, n, p, q])),
            tensor_type(scope='global', dtype='float32', layout=DataLayout.row_major([m, n, p, q]))
        ],
        worker=Grid()
    )


def log_softmax(data_shape, axis):
    pass

#
#  task=CustomOp(tensor_pattern, tag='softmax')
#
#  pattern CustomOpPattern()
#
