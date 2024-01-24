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
from typing import List, Optional, Sequence, Tuple, Union
from hidet.ir.builders import StmtBuilder
from hidet.ir.primitives import active_mask, shfl_down_sync, shfl_sync
from hidet.ir.stmt import AssignStmt, Stmt
from hidet.utils import gcd, prod
from hidet.ir.mapping import TaskMapping, row_repeat, spatial_map
from hidet.ir.layout import DataLayout, row_major, local_layout
from hidet.ir.expr import Expr, is_true
from hidet.ir.dtypes import f16, f32
from hidet.ir.type import DataType
from hidet.cuda.cublas import cublasComputeType


class NotSupportedError(Exception):
    def __init__(self, obj: object, msg: str = ""):
        super().__init__()
        self.obj = obj
        self.msg = msg


def warp_reduce(v, op) -> Stmt:
    """
    Reduce over the threads in a warp.

    Parameters
    ----------
    v: Var
        The value to reduce. It must be a variable.
    op:
        An binary operator to represent the reducing operator, must be communicative and associative.

    Returns
    -------
    ret: Stmt
        A block statement to finish the reduction. After reduction, the value in each thread in the warp
        has the reduced value.
    """
    sb = StmtBuilder()
    with sb.let('mask', active_mask()) as mask:
        for delta in [16, 8, 4, 2, 1]:
            sb += AssignStmt(v, op(v, shfl_down_sync(mask, v, delta=delta)))
        sb += AssignStmt(v, shfl_sync(mask, v, src_lane=0))
    return sb.finish()


def _get_shapes(
    task_shape: Sequence[int], num_workers=32, perm: Optional[Sequence[int]] = None
) -> Tuple[List[int], List[int]]:
    rank = len(task_shape)

    if prod(task_shape) % num_workers != 0:
        raise NotSupportedError(
            'Number of workers must be a divisor of total number of tasks, '
            'task shape {} and number workers {}.'.format(task_shape, num_workers)
        )

    # can not have duplicated dimensions
    if len(set(perm)) != len(perm):
        raise NotSupportedError('Duplicated ranks in perm: {}'.format(perm))

    if len(perm) != rank:
        raise NotSupportedError('Length of perm {} does not match task_shape {}'.format(perm, task_shape))

    if len(set(perm) - set(range(rank))) != 0:
        raise NotSupportedError('perm should be a permutation of {}, got {}'.format(list(range(rank)), perm))

    # first fill in grid_shape
    grid_shape = [0 for _ in range(rank)]
    for i in reversed(range(rank)):
        dim = perm.index(i)
        factor = gcd(task_shape[dim], num_workers)
        grid_shape[dim] = factor
        num_workers //= factor
    assert num_workers == 1

    # remaining tasks are repeated by workers
    repeat_shape = []
    for dim in range(rank):
        repeat_shape.append(task_shape[dim] // grid_shape[dim])

    return grid_shape, repeat_shape


def get_task_map(task_shape: Sequence[int], num_workers=32, ranks: Sequence[int] = None) -> TaskMapping:
    """
    Get a task map that maps a collection of workers to a task domain with given shape. The returned
    task map is composed of repeat shape and grid shape. We first determine the size of each dimension
    in the grid shape, then fill the repeat shape accordingly.

    It follows the following steps to construct the task map.
        1. Normalize the order of dimensions. The last dimension in the order is continuous regarding
           worker index.
        2. Following the order of dimension, determine the grid shape.
        2. Fill the repeat shape.

    Parameters
    ----------
    task_shape: Sequence[int]
        The shape of the task domain.
    num_workers: int
        The number of workers.
    ranks: Optional[Sequence[int]]
        todo: finish this.

    Returns
    -------
    ret: TaskMapping
        The task mapping that maps given number of workers to given task domain.

    Examples
    --------

    >>> get_task_map([4, 4], num_workers=2, ranks=[0, 1])
    [[0 1 0 1]
     [0 1 0 1]
     [0 1 0 1]
     [0 1 0 1]]

    >>> get_task_map([4, 4], num_workers=2, ranks=[1, 0])
    [[0 0 0 0]
     [1 1 1 1]
     [0 0 0 0]
     [1 1 1 1]]
    """
    grid_shape, repeat_shape = _get_shapes(task_shape, num_workers, ranks)

    task_map = row_repeat(*repeat_shape) * spatial_map(grid_shape, ranks=ranks)
    return task_map


def get_transfer_task_map(
    task_shape: Sequence[int], num_workers=32, ranks: Optional[Sequence[int]] = None
) -> Tuple[TaskMapping, DataLayout]:
    grid_shape, repeat_shape = _get_shapes(task_shape, num_workers, ranks)

    task_map = row_repeat(*repeat_shape, attrs='u+u+') * spatial_map(grid_shape, ranks=ranks)
    data_layout = row_major(*repeat_shape) * local_layout(*grid_shape)
    return task_map, data_layout


def resolve_cublas_compute_type(
    in_dtype: DataType, out_dtype: DataType, compute_type: Optional[Union[int, cublasComputeType]]
) -> cublasComputeType:
    if compute_type is not None:
        return cublasComputeType(compute_type)
    if in_dtype == out_dtype == f16:
        # use tensor core whenever possible
        return cublasComputeType.CUBLAS_COMPUTE_16F
    elif in_dtype == out_dtype == f32:
        # use tensor core whenever possible
        return cublasComputeType.CUBLAS_COMPUTE_32F
    else:
        raise NotImplementedError(
            'not implemented resolve rules for compute_type with in_dtype={}, out_dtype={}'.format(in_dtype, out_dtype)
        )


def convert_to_cublas_strided_gemm(a_shape: List[Expr], b_shape: List[Expr], c_shape: List[Expr]):
    a_rank: int = len(a_shape)
    b_rank: int = len(b_shape)

    assert a_rank >= 1 and b_rank >= 1 and (a_rank >= 2 or b_rank >= 2)
    if a_rank == 1:
        bs = prod(b_shape[:-2])
        m = 1
        n = b_shape[-1]
        k = a_shape[0]
        stride_a = 0
        stride_b = b_shape[-2] * b_shape[-1]
        stride_c = c_shape[-2] * c_shape[-1]
    elif b_rank == 1:
        bs = prod(a_shape[:-2])
        m = a_shape[-2]
        n = 1
        k = b_shape[0]
        stride_a = a_shape[-2] * a_shape[-1]
        stride_b = 0
        stride_c = c_shape[-1]
    else:
        if is_true(prod(a_shape[:-2]) == 1):
            bs = prod(b_shape[:-2])
            m = a_shape[-2]
            n = b_shape[-1]
            k = a_shape[-1]
            stride_a = 0
            stride_b = b_shape[-2] * b_shape[-1]
            stride_c = c_shape[-2] * c_shape[-1]
        elif is_true(prod(b_shape[:-2]) == 1):
            bs = prod(a_shape[:-2])
            m = a_shape[-2]
            n = b_shape[-1]
            k = a_shape[-1]
            stride_a = a_shape[-2] * a_shape[-1]
            stride_b = 0
            stride_c = c_shape[-2] * c_shape[-1]
        elif all(is_true(a == b) for a, b in zip(a_shape[:-2], b_shape[:-2])):
            bs = prod(a_shape[:-2])
            m = a_shape[-2]
            n = b_shape[-1]
            k = a_shape[-1]
            stride_a = a_shape[-2] * a_shape[-1]
            stride_b = b_shape[-2] * b_shape[-1]
            stride_c = c_shape[-2] * c_shape[-1]
        else:
            # todo: add cublasGemmBatchedEx to support this case
            # https://docs.nvidia.com/cuda/cublas/index.html#cublasgemmbatchedex
            raise NotImplementedError('Can not convert matmul {} @ {} to strided_gemm'.format(a_shape, b_shape))
    return bs, m, n, k, stride_a, stride_b, stride_c


def get_cublas_matmul_schedule(a_shape, b_shape, c_shape, a_dtype, b_dtype, c_dtype):
    import hidet
    from hidet.lang.cuda import cublas
    from hidet.lang import attrs
    from hidet.ir.library import tune

    try:
        bs, m, n, k, stride_a, stride_b, stride_c = convert_to_cublas_strided_gemm(a_shape, b_shape, c_shape)
        compute_type: cublasComputeType = resolve_cublas_compute_type(a_dtype, b_dtype, None)
    except NotImplementedError:
        # Unable to resolve cublas params, skip using cublas
        tune.check(False)

    with hidet.script_module() as script_module:

        def generate(a: Expr, b: Expr, c: Expr) -> Expr:
            return cublas.strided_gemm(
                bs,
                m,
                n,
                k,
                a_dtype,
                b_dtype,
                c_dtype,
                a,
                b,
                c,
                stride_a,
                stride_b,
                stride_c,
                False,
                False,
                compute_type,
            )

        @hidet.script
        def launch(a: a_dtype[a_shape], b: b_dtype[b_shape], c: c_dtype[c_shape]):
            attrs.func_kind = 'public'

            generate(a, b, c)

    return script_module.ir_module()
