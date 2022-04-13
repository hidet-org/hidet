from typing import Union, Sequence, List
from ..common import Task, Operator, Tensor, DataLayout, TensorInput, Grid, tensor_input, compute, reduce, inline_compute, tensor_type, input_like
from hidet.ir.primitives import expf


def softmax_task(x: TensorInput, axis: int):
    shape = x.const_shape()
    axis = (axis + len(shape)) % len(shape)
    if not (len(shape) == 4 and axis == 1):
        # todo: support other cases
        msg = 'Currently, only support softmax a tensor with 4 dimensions along the second dimension. ' + \
              'Please use .flatten() and .unsqueeze() to transform the input first.'
        raise NotImplementedError(msg)
    m, n, p, q = shape
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
        fcompute=lambda i, j, r, s: e[i, j, r, s] / se[i, r, s],
        scope='global'
    )
    return Task(
        name='softmax',
        computation=out,
        params=[x, out],
        worker=Grid()
    )


class SoftmaxOp(Operator):
    def __init__(self,
                 x: Tensor,
                 axis: int = 1):
        super().__init__(
            inputs=[x],
            task=softmax_task(input_like(x, 'x'), axis),
            axis=axis
        )


def softmax(x: Tensor, axis=1) -> Tensor:
    import hidet.tos.operators as ops
    if len(x.shape) < 4:
        dims = list(range(len(x.shape), 4))
        xx = ops.unsqueeze(x, dims)
        return SoftmaxOp(xx, axis).get_output(0).squeeze(dims)
    return SoftmaxOp(x, axis).get_output(0)

