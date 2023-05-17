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
    def __init__(self, task_mapping: TaskMapping, worker):
        super().__init__()
        self.task_mapping: TaskMapping = task_mapping
        self.worker: Expr = worker

    def __iter__(self):
        raise NotImplementedError("TaskMappingLoopIterable is not iterable for now.")

    def generate_loop_statement(self, loop_vars: List[Var], body: Stmt) -> Stmt:
        assert len(loop_vars) == len(self.task_mapping.task_shape)
        return ForMappingStmt(loop_vars, self.task_mapping, self.worker, body)

    def num_loop_vars(self) -> int:
        return len(self.task_mapping.task_shape)


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


def grid(*dim_extents, attrs: Optional[str] = None):
    """
    Iterate over the grid.

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
    bind_tuple = False
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
