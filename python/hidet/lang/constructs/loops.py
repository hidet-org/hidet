from typing import List, Optional, Sequence, Union
from hidet.ir.expr import Var, Expr
from hidet.ir.stmt import Stmt, ForStmtAttr, ForStmt, ForMappingStmt
from hidet.ir.mapping import TaskMapping


class HidetLoopIterable:
    def generate_loop_statement(self, loop_vars: List[Var], body: Stmt) -> Stmt:
        raise NotImplementedError()

    def num_loop_vars(self) -> int:
        raise NotImplementedError()


class TaskMappingLoopIterable(HidetLoopIterable):
    def __init__(self, task_mapping: TaskMapping, worker):
        super().__init__()
        self.task_mapping: TaskMapping = task_mapping
        self.worker: Expr = worker

    def generate_loop_statement(self, loop_vars: List[Var], body: Stmt) -> Stmt:
        assert len(loop_vars) == len(self.task_mapping.task_shape)
        return ForMappingStmt(loop_vars, self.task_mapping, self.worker, body)

    def num_loop_vars(self) -> int:
        return len(self.task_mapping.task_shape)


class GridLoopIterable(HidetLoopIterable):
    def __init__(self, extents: Sequence[Union[Expr, int]], attrs: Optional[str]):
        super().__init__()
        self.extents: List[Expr] = list(extents)
        self.attrs: List[ForStmtAttr] = ForStmtAttr.parse(attrs, len(self.extents))

    def generate_loop_statement(self, loop_vars: List[Var], body: Stmt) -> Stmt:
        assert len(loop_vars) == len(self.extents)
        for var, extent, attr in reversed(zip(loop_vars, self.extents, self.attrs)):
            body = ForStmt(var, extent, body, attr=attr)
        return body

    def num_loop_vars(self) -> int:
        return len(self.extents)


class RangeLoopIterable(HidetLoopIterable):
    def __init__(self, end: Union[Expr, int]):
        super().__init__()
        self.end = end

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
    dim_extents: Sequence[Expr or int or str]
        The length of each dimension. The last one can be the attrs.

    attrs: Optional[str]
        The attributes of each loop. See hidet.stmt.ForStmtAttr for more information.

    Returns
    -------
    indices: Sequence[Tuple[Expr, ...]]
        The sequence of indices in the grid to be iterated. (This is the semantics of hidet script, not this python
        function.)
    """
    return GridLoopIterable(dim_extents, attrs)


def range(end: Union[Expr, int], real_end: Optional[Union[Expr, int]] = None, step: Union[Expr, int] = 1):
    if real_end is not None or step != 1:
        raise NotImplementedError('Currently, we only support range(end) instead of range(start, end, step).')
    return RangeLoopIterable(end)
