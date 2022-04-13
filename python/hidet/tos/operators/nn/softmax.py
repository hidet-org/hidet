from typing import Union, Sequence, List
from ..common import Task, Operator, Tensor, DataLayout, TensorInput, Grid, tensor_input, compute, reduce, inline_compute, tensor_type, input_like, normalize_dim, custom_compute
from hidet.ir.primitives import expf


def softmax_task_stale(x: TensorInput, axis: int):
    shape = x.const_shape()
    axis = (axis + len(shape)) % len(shape)
    if not (len(shape) == 4 and axis == 1):
        # todo: support other cases
        msg = 'Currently, only support softmax a tensor with 4 dimensions along the second dimension. \n' + \
              'Got x shape {}, axis {}.\n'.format(x.const_shape(), axis) + \
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


def softmax_task(x: TensorInput, axis: int) -> Task:
    out = custom_compute(
        name='out',
        identifier='softmax',
        params=[x],
        data_type=x.data_type,
        attributes={
            'axis': axis
        }
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
        axis = normalize_dim(axis, len(x.shape))
        super().__init__(
            inputs=[x],
            task=softmax_task(input_like(x, 'x'), axis),
            axis=axis
        )


def softmax(x: Tensor, axis=1) -> Tensor:
    import hidet.tos.operators as ops
    # if len(x.shape) < 4:
    #     dims = list(range(len(x.shape), 4))
    #     xx = ops.unsqueeze(x, dims)
    #     return SoftmaxOp(xx, axis).get_output(0).squeeze(dims)
    # if axis > 1:
    #     x = x.flatten(start_dim=0, end_dim=axis)
    #     axis = 1
    #     if len(x.shape) > 4:
    #         x = x.flatten(start_dim=2)
    #     if len(x.shape) < 4:
    #         x = x.unsqueeze(dims=range(len(x.shape), 4))
    return SoftmaxOp(x, axis).get_output(0)

