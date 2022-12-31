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
from typing import List, Optional, Sequence, Tuple
from hidet.ir.builders import StmtBuilder
from hidet.ir.primitives import active_mask, shfl_down_sync, shfl_sync
from hidet.ir.stmt import AssignStmt, Stmt
from hidet.utils import gcd, prod
from hidet.ir.mapping import TaskMapping, row_repeat, spatial_map
from hidet.ir.layout import DataLayout, row_layout, local_layout
from hidet.graph.ops.schedules.common import NotSupportedError


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

    task_map = row_repeat(*repeat_shape) * spatial_map(grid_shape, ranks=ranks)
    data_layout = row_layout(*repeat_shape) * local_layout(*grid_shape)
    return task_map, data_layout
