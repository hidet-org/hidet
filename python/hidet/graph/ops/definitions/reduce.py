from typing import List, Union

from .arithmetic import square, sqrt
from .utils import Task, Operator, Tensor, TensorNode, IRModule, compute, reduce, input_like, normalize_dim, arg_reduce


class ReduceTask(Task):
    def __init__(
        self, x: TensorNode, dims: List[int], keep_dim: bool, reduce_type: str, accumulate_dtype: str = 'float32'
    ):
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
            return reduce(
                shape=reduce_shape, fcompute=reduce_fcompute, reduce_type=reduce_type, accumulate_dtype=accumulate_dtype
            )

        y = compute(name='y', shape=y_shape, fcompute=fcompute)

        self.dims: List[int] = dims
        self.keep_dim: bool = keep_dim
        self.reduce_type: str = reduce_type

        super().__init__(
            name='reduce_{}'.format(reduce_type),
            inputs=[x],
            outputs=[y],
            attributes={
                'dims': dims,
                'keep_dim': keep_dim,
                'reduce_type': reduce_type,
                'accumulate_dtype': accumulate_dtype,
            },
        )

    def implement_cuda(self, workding_dir: str) -> IRModule:
        # pylint: disable=import-outside-toplevel
        from ..schedules import cuda_schedule_reduce_by_default, cuda_schedule_reduce_by_warp_reduce

        rank = len(self.inputs[0].const_shape())
        if rank - 1 in self.dims:
            # reduce over last dimension
            return cuda_schedule_reduce_by_warp_reduce(self)
        else:
            # last dimension has not been reduced
            return cuda_schedule_reduce_by_default(self)


class ArgReduceTask(Task):
    def __init__(self, x: TensorNode, dim: int, keep_dim: bool, reduce_type: str):
        x_shape = x.const_shape()
        y_shape = []
        for i, extent in enumerate(x_shape):
            if i == dim:
                if keep_dim:
                    y_shape.append(1)
            else:
                y_shape.append(extent)

        def fcompute(*indices):
            def reduce_fcompute(reduce_index):
                x_indices = indices[:dim] + (reduce_index,) + indices[dim + (1 if keep_dim else 0) :]
                return x[x_indices]

            return arg_reduce(
                extent=x_shape[dim], fcompute=reduce_fcompute, reduce_type=reduce_type, index_dtype='int32'
            )

        y = compute(name='y', shape=y_shape, fcompute=fcompute)
        super().__init__(
            name='arg{}'.format(reduce_type),
            inputs=[x],
            outputs=[y],
            attributes={'dim': dim, 'reduce_type': reduce_type},
        )


class ReduceBaseOp(Operator):
    def __init__(self, x: Tensor, dims: List[int], keep_dim: bool, reduce_type: str):
        if reduce_type not in ['avg', 'max', 'min', 'sum']:
            raise NotImplementedError('Do not support reduce type: {}'.format(reduce_type))
        dims = normalize_dim(dims, rank=len(x.shape))
        super().__init__(
            inputs=[x],
            task=ReduceTask(input_like(x, 'x'), dims, keep_dim, reduce_type),
            attributes={'dims': dims, 'keep_dim': keep_dim},
        )


class ArgReduceBaseOp(Operator):
    def __init__(self, x: Tensor, dim: int, keep_dim: bool, reduce_type: str):
        if reduce_type not in ['min', 'max']:
            raise NotImplementedError('Do not support reduce type: {}'.format(reduce_type))
        super().__init__(
            inputs=[x],
            task=ArgReduceTask(input_like(x, 'x'), dim, keep_dim, reduce_type),
            attributes={'dim': dim, 'keep_dim': keep_dim},
        )


class ReduceMeanOp(ReduceBaseOp):
    def __init__(self, x: Tensor, dims: List[int], keep_dim: bool = False):
        super().__init__(x, dims, keep_dim, 'avg')


class ReduceSumOp(ReduceBaseOp):
    def __init__(self, x: Tensor, dims: List[int], keep_dim: bool = False):
        super().__init__(x, dims, keep_dim, 'sum')


class ReduceMaxOp(ReduceBaseOp):
    def __init__(self, x: Tensor, dims: List[int], keep_dim: bool = False):
        super().__init__(x, dims, keep_dim, 'max')


class ReduceMinOp(ReduceBaseOp):
    def __init__(self, x: Tensor, dims: List[int], keep_dim: bool = False):
        super().__init__(x, dims, keep_dim, 'min')


class ReduceProdOp(ReduceBaseOp):
    def __init__(self, x: Tensor, dims: List[int], keep_dim: bool = False):
        super().__init__(x, dims, keep_dim, 'prod')


class ArgMinOp(ArgReduceBaseOp):
    def __init__(self, x: Tensor, dim: int, keep_dim: bool):
        super().__init__(x, dim, keep_dim, 'min')


class ArgMaxOp(ArgReduceBaseOp):
    def __init__(self, x: Tensor, dim: int, keep_dim: bool):
        super().__init__(x, dim, keep_dim, 'max')


def mean(x: Tensor, dims: Union[int, List[int]], keep_dim: bool = False) -> Tensor:
    if isinstance(dims, int):
        dims = [dims]
    return ReduceMeanOp(x, dims, keep_dim).get_output(0)


def sum(x: Tensor, dims: Union[int, List[int]], keep_dim: bool = False) -> Tensor:
    if isinstance(dims, int):
        dims = [dims]
    return ReduceSumOp(x, dims, keep_dim).get_output(0)


def max(x: Tensor, dims: Union[int, List[int]], keep_dim: bool = False) -> Tensor:
    if isinstance(dims, int):
        dims = [dims]
    return ReduceMaxOp(x, dims, keep_dim).get_output(0)


def min(x: Tensor, dims: Union[int, List[int]], keep_dim: bool = False) -> Tensor:
    if isinstance(dims, int):
        dims = [dims]
    return ReduceMinOp(x, dims, keep_dim).get_output(0)


def var(x: Tensor, dims: Union[int, List[int]], keep_dim: bool = False) -> Tensor:
    x = x - x.mean(dims=dims, keep_dim=True)
    return square(x).mean(dims=dims, keep_dim=keep_dim)


def std(x: Tensor, dims: Union[int, List[int]], keep_dim: bool = False) -> Tensor:
    return sqrt(var(x, dims=dims, keep_dim=keep_dim))


def prod(x: Tensor, dims: Union[int, List[int]], keep_dim: bool = False) -> Tensor:
    if isinstance(dims, int):
        dims = [dims]
    return ReduceProdOp(x, dims, keep_dim).get_output(0)


def argmin(x: Tensor, dim: int, keep_dim: bool = False) -> Tensor:
    return ArgMinOp(x, dim, keep_dim).get_output(0)


def argmax(x: Tensor, dim: int, keep_dim: bool = False) -> Tensor:
    return ArgMaxOp(x, dim, keep_dim).get_output(0)


def all(x: Tensor, /, *, axis=None, keepdims=False) -> Tensor:
    raise NotImplementedError()


def any(x: Tensor, /, *, axis=None, keepdims=False) -> Tensor:
    raise NotImplementedError()
