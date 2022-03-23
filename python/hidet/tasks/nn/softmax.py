from typing import List
from hidet.ir.dialects.compute import tensor_input, compute, reduce
from hidet.ir.layout import DataLayout
from hidet.ir.task import Task, Grid
from hidet.ir.type import tensor_type
from hidet.ir.functors import inline_compute
from hidet.ir.primitives import expf


def softmax(shape: List[int], axis: int):
    # todo: support other shape
    assert len(shape) == 2
    assert axis == 1
    m, n = shape
    x = tensor_input('x', 'float32', shape=[m, n])
    mx = compute(
        name='mx',
        shape=[m],
        fcompute=lambda i: reduce(
            shape=[n],
            fcompute=lambda j: x[i, j],
            reduce_type='max'
        )
    )
    e = compute(
        name='e',
        shape=[m, n],
        fcompute=lambda i, j: expf(x[i, j] - mx[i])
    )
    se = compute(
        name='se',
        shape=[m],
        fcompute=lambda i: reduce(
            shape=[n],
            fcompute=lambda j: e[i, m],
            reduce_type='sum'
        )
    )
    out = compute(
        name='out',
        shape=[m, n],
        fcompute=lambda i, j: e[i, j] / se[i]
    )
    return Task(
        name='softmax',
        computation=out,
        params=[x, out],
        params_type=[
            tensor_type(scope='global', dtype='float32', layout=DataLayout.row_major([m, n])),
            tensor_type(scope='global', dtype='float32', layout=DataLayout.row_major([m, n]))
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
