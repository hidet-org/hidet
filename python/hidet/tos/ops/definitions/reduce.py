from typing import List, Union

from .arithmatic import square
from .utils import Task, Operator, Tensor, TensorInput, compute, reduce, input_like, normalize_dim


class ReduceTask(Task):
    def __init__(self, x: TensorInput, dims: List[int], keep_dim: bool, reduce_type: str):
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
            return reduce(shape=reduce_shape, fcompute=reduce_fcompute, reduce_type=reduce_type)

        y = compute(name='y', shape=y_shape, fcompute=fcompute, scope='global')
        super().__init__(
            name='reduce_{}'.format(reduce_type),
            inputs=[x],
            outputs=[y],
        )


class ReduceMeanOp(Operator):
    def __init__(self, x: Tensor, dims: List[int], keep_dim: bool = False):
        dims = normalize_dim(dims, rank=len(x.shape))
        super().__init__(
            inputs=[x],
            task=ReduceTask(input_like(x, 'x'), dims, keep_dim, 'avg'),
            dims=dims,
            keep_dim=keep_dim
        )


class ReduceSumOp(Operator):
    def __init__(self, x: Tensor, dims: List[int], keep_dim: bool = False):
        dims = normalize_dim(dims, rank=len(x.shape))
        super().__init__(
            inputs=[x],
            task=ReduceTask(input_like(x, 'x'), dims, keep_dim, 'sum'),
            dims=dims,
            keep_dim=keep_dim
        )


def reduce_mean(x: Tensor, dims: Union[int, List[int]], keep_dim: bool = False) -> Tensor:
    if isinstance(dims, int):
        dims = [dims]
    return ReduceMeanOp(x, dims, keep_dim).get_output(0)


def reduce_sum(x: Tensor, dims: Union[int, List[int]], keep_dim: bool = False) -> Tensor:
    if isinstance(dims, int):
        dims = [dims]
    return ReduceSumOp(x, dims, keep_dim).get_output(0)


def reduce_var(x: Tensor, dims: Union[int, List[int]], keep_dim: bool = False) -> Tensor:
    return reduce_mean(square(x - reduce_mean(x, dims, keep_dim=True)), dims, keep_dim)

