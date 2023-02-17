# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import List, Union, Optional, Sequence

from ..arithmetic import square, sqrt
from ..utils import Task, Operator, Tensor, TensorNode, IRModule, ReduceType
from ..utils import compute, reduce, input_like, normalize_dim, arg_reduce


class ReduceTask(Task):
    def __init__(
        self, x: TensorNode, dims: List[int], keep_dim: bool, reduce_type: ReduceType, accumulate_dtype: str = 'float32'
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
        self.reduce_type: ReduceType = reduce_type

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
        from ...schedules import cuda_schedule_reduce_by_default, cuda_schedule_reduce_by_warp_reduce

        rank = len(self.inputs[0].const_shape())
        if rank - 1 in self.dims:
            # reduce over last dimension
            return cuda_schedule_reduce_by_warp_reduce(self)
        else:
            # last dimension has not been reduced
            return cuda_schedule_reduce_by_default(self)


class ArgReduceTask(Task):
    def __init__(self, x: TensorNode, dim: int, keep_dim: bool, reduce_type: ReduceType):
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
                extent=x_shape[dim], fcompute=reduce_fcompute, reduce_type=reduce_type, index_dtype='int64'
            )

        y = compute(name='y', shape=y_shape, fcompute=fcompute)
        super().__init__(
            name='arg{}'.format(reduce_type),
            inputs=[x],
            outputs=[y],
            attributes={'dim': dim, 'reduce_type': reduce_type},
        )


class ReduceBaseOp(Operator):
    def __init__(self, x: Tensor, dims: Optional[Sequence[int]], keep_dim: bool, reduce_type: ReduceType):
        rank = len(x.shape)
        if dims is None:
            dims = list(range(rank))
        dims = normalize_dim(dims, rank=rank)
        super().__init__(
            inputs=[x],
            task=ReduceTask(input_like(x, 'x'), dims, keep_dim, reduce_type),
            attributes={'dims': dims, 'keepdims': keep_dim},
        )


class ArgReduceBaseOp(Operator):
    def __init__(self, x: Tensor, dim: int, keep_dim: bool, reduce_type: ReduceType):
        if reduce_type not in [ReduceType.Min, ReduceType.Max]:
            raise NotImplementedError('Do not support arg reduce type: {}'.format(reduce_type))
        super().__init__(
            inputs=[x],
            task=ArgReduceTask(input_like(x, 'x'), dim, keep_dim, reduce_type),
            attributes={'dim': dim, 'keepdims': keep_dim},
        )


class ReduceMeanOp(ReduceBaseOp):
    def __init__(self, x: Tensor, dims: Optional[Sequence[int]], keepdims: bool = False):
        super().__init__(x, dims, keepdims, ReduceType.Average)


class ReduceSumOp(ReduceBaseOp):
    def __init__(self, x: Tensor, dims: Optional[Sequence[int]], keepdims: bool = False):
        super().__init__(x, dims, keepdims, ReduceType.Sum)


class ReduceMaxOp(ReduceBaseOp):
    def __init__(self, x: Tensor, dims: Optional[Sequence[int]], keepdims: bool = False):
        super().__init__(x, dims, keepdims, ReduceType.Max)


class ReduceMinOp(ReduceBaseOp):
    def __init__(self, x: Tensor, dims: Optional[Sequence[int]], keepdims: bool = False):
        super().__init__(x, dims, keepdims, ReduceType.Min)


class ReduceOrOp(ReduceBaseOp):
    def __init__(self, x: Tensor, dims: Optional[Sequence[int]], keepdims: bool = False):
        super().__init__(x, dims, keepdims, ReduceType.Or)


class ReduceAndOp(ReduceBaseOp):
    def __init__(self, x: Tensor, dims: Optional[Sequence[int]], keepdims: bool = False):
        super().__init__(x, dims, keepdims, ReduceType.And)


class ReduceProdOp(ReduceBaseOp):
    def __init__(self, x: Tensor, dims: Optional[Sequence[int]], keepdims: bool = False):
        super().__init__(x, dims, keepdims, ReduceType.Product)


class ArgMinOp(ArgReduceBaseOp):
    def __init__(self, x: Tensor, dim: int, keepdims: bool):
        super().__init__(x, dim, keepdims, ReduceType.Min)


class ArgMaxOp(ArgReduceBaseOp):
    def __init__(self, x: Tensor, dim: int, keepdims: bool):
        super().__init__(x, dim, keepdims, ReduceType.Max)


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
    """
    Check if all of the elements on the given axis evaluates to True.

    Parameters
    ----------
    x: Tensor
        The input tensor.

    axis: int or Sequence[int], optional
        The axis or axes along which to perform the logical AND. None indicates to perform the reduction on the whole
        tensor. When an integer or a sequence of integers are given, they must be in range [-N, N), where N is the
        rank of the input tensor.

    keepdims: bool, default=False
        Whehter to keep the dimension.

    Returns
    -------
    ret: Tensor
        The result of logical AND reduction with bool data type.
    """
    x = x.astype('bool')
    return ReduceAndOp(x, dims=axis, keepdims=keepdims).get_output(0)


def any(x: Tensor, /, *, axis=None, keepdims=False) -> Tensor:
    """
    Check if any of the elements on the given axis evaluates to True.

    Parameters
    ----------
    x: Tensor
        The input tensor.

    axis: int or Sequence[int], optional
        The axis or axes along which to perform the logical OR. None indicates to perform the reduction on the whole
        tensor. When an integer or a sequence of integers are given, they must be in range [-N, N), where N is the
        rank of the input tensor.

    keepdims: bool, default=False
        Whehter to keep the dimension.

    Returns
    -------
    ret: Tensor
        The result of logical OR reduction with bool data type.
    """
    x = x.astype('bool')
    return ReduceOrOp(x, dims=axis, keepdims=keepdims).get_output(0)
