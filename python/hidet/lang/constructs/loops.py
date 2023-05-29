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
from typing import List, Optional, Sequence, Union
import itertools
import builtins
from hidet.ir.expr import Var, Expr, Constant
from hidet.ir.stmt import Stmt, ForStmtAttr, ForStmt, ForMappingStmt
from hidet.ir.mapping import TaskMapping


class HidetLoopIterable:
    def generate_loop_statement(self, loop_vars: List[Var], body: Stmt) -> Stmt:
        raise NotImplementedError()

    def num_loop_vars(self) -> int:
        raise NotImplementedError()

    def bind_tuple(self) -> bool:
        return False

    @staticmethod
    def extract_int(extent: Union[Expr, int]) -> int:
        if isinstance(extent, int):
            return extent
        elif isinstance(extent, Constant) and extent.type.is_data_type() and extent.type.is_integer():
            return int(extent)
        else:
            raise ValueError('end must be an integer or a constant integer.')


class TaskMappingLoopIterable(HidetLoopIterable):
    def __init__(self, task_mapping: TaskMapping, worker, bind_tuple=False):
        super().__init__()
        self.task_mapping: TaskMapping = task_mapping
        self.worker: Expr = worker
        self._bind_tuple: bool = bind_tuple

    def __iter__(self):
        return iter(self.task_mapping.worker2task(self.worker))

    def generate_loop_statement(self, loop_vars: List[Var], body: Stmt) -> Stmt:
        assert len(loop_vars) == len(self.task_mapping.task_shape)
        return ForMappingStmt(loop_vars, self.task_mapping, self.worker, body)

    def num_loop_vars(self) -> int:
        return len(self.task_mapping.task_shape)

    def bind_tuple(self) -> bool:
        return self._bind_tuple


class GridLoopIterable(HidetLoopIterable):
    def __init__(self, extents: Sequence[Union[Expr, int]], attrs: Optional[str], bind_tuple: bool):
        super().__init__()
        self.extents: List[Expr] = list(extents)
        self.attrs: List[ForStmtAttr] = ForStmtAttr.parse(attrs, len(self.extents))
        self._bind_tuple = bind_tuple

    def __iter__(self):
        extents = [self.extract_int(extent) for extent in self.extents]
        return itertools.product(*[range(extent) for extent in extents]).__iter__()

    def generate_loop_statement(self, loop_vars: List[Var], body: Stmt) -> Stmt:
        assert len(loop_vars) == len(self.extents)
        for var, extent, attr in reversed(list(zip(loop_vars, self.extents, self.attrs))):
            body = ForStmt(var, extent, body, attr=attr)
        return body

    def num_loop_vars(self) -> int:
        return len(self.extents)

    def bind_tuple(self) -> bool:
        return self._bind_tuple


class RangeLoopIterable(HidetLoopIterable):
    def __init__(self, end: Union[Expr, int]):
        super().__init__()
        self.end = end

    def __iter__(self):
        return builtins.range(self.extract_int(self.end)).__iter__()

    def generate_loop_statement(self, loop_vars: List[Var], body: Stmt) -> Stmt:
        assert len(loop_vars) == 1
        return ForStmt(loop_vars[0], self.end, body)

    def num_loop_vars(self) -> int:
        return 1


def grid(*dim_extents, attrs: Optional[str] = None, bind_tuple=False):
    """
    Iterate over the grid.

    Usage 1: specify the grid dimensions with positional arguments.
      for i, j in grid(2, 3):
          printf("%d %d\n", i, j)

      for indices in grid(2, 3):
          printf("%d %d\n", indices[0], indices[1])

      for i in grid(2):
          printf("%d\n", i)

    Usage 2: specify the grid dimensions with a list or tuple.
      for indices in grid([2, 3]):
          printf("%d %d\n", indices[0], indices[1])

      for indices in grid([2]):  # indices is a tuple with one element
          printf("%d %d\n", indices[0])

      for indices in grid(2, bind_tuple=True):  # indices is a tuple with one element
          printf("%d %d\n", indices[0])

    Usage 3: specify the loop attribute
      for i, j in grid(2, 3, attrs='up'):   # loop i is unrolled while loop j is parallelized
          printf("%d %d\n", i, j)

    Parameters
    ----------
    dim_extents: Sequence[Expr or int or list or tuple or str]
        The length of each dimension. The last one can be the attrs.

    attrs: Optional[str]
        The attributes of each loop. See hidet.stmt.ForStmtAttr for more information.

    Returns
    -------
    indices: Sequence[Tuple[Expr, ...]]
        The sequence of indices in the grid to be iterated. (This is the semantics of hidet script, not this python
        function.)
    """
    if len(dim_extents) == 1 and isinstance(dim_extents[0], (list, tuple)):
        dim_extents = dim_extents[0]
        bind_tuple = True
    if len(dim_extents) > 1 and isinstance(dim_extents[-1], str):
        if attrs is not None:
            raise ValueError('attrs cannot be specified twice.')
        attrs = dim_extents[-1]
        dim_extents = dim_extents[:-1]
    return GridLoopIterable(dim_extents, attrs, bind_tuple)


def range(end: Union[Expr, int], real_end: Optional[Union[Expr, int]] = None, step: Union[Expr, int] = 1):
    if real_end is not None or step != 1:
        raise NotImplementedError('Currently, we only support range(end) instead of range(start, end, step).')
    return RangeLoopIterable(end)
