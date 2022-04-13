from typing import List

from ..common import Task, Operator, Tensor, TensorInput, Grid, compute, reduce, input_like, normalize_dim


def reduce_mean_task(x: TensorInput, dims: List[int], keep_dim=False):
    x_shape = x.const_shape()
    y_shape = []
    for i in range(len(x_shape)):
        if i in dims:
            if keep_dim:
                y_shape.append(1)
        else:
            y_shape.append(x_shape[i])

    def fcompute(*indices):
        def reduce_fcompute(*reduce_indices):
            x_indices = []
            p = 0
            q = 0
            for i in range(len(x_shape)):
                if i not in dims:
                    x_indices.append(indices[p])
                    p += 1
                else:
                    x_indices.append(reduce_indices[q])
                    q += 1
                    if keep_dim:
                        p += 1
            assert p == len(indices) and q == len(reduce_indices)
            return x[x_indices]

        reduce_shape = [x_shape[i] for i in dims]
        return reduce(shape=reduce_shape, fcompute=reduce_fcompute, reduce_type='avg')

    y = compute(name='y', shape=y_shape, fcompute=fcompute, scope='global')
    return Task(
        'reduce_mean',
        computation=y,
        params=[x, y],
        worker=Grid()
    )


class ReduceMeanOp(Operator):
    def __init__(self, x: Tensor, dims: List[int], keep_dim: bool = False):
        dims = normalize_dim(dims, rank=len(x.shape))
        super().__init__(
            inputs=[x],
            task=reduce_mean_task(input_like(x, 'x'), dims, keep_dim),
            dims=dims,
            keep_dim=keep_dim
        )


def reduce_mean(x: Tensor, dims: List[int], keep_dim: bool = False) -> Tensor:
    return ReduceMeanOp(x, dims, keep_dim).get_output(0)
