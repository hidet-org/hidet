from typing import Union, Optional, Sequence, List

from hidet.ir.stmt import Stmt, ForStmt, IfStmt, LetStmt, EvaluateStmt, SeqStmt
from hidet.ir.type import TypeNode, scalar_type, ScalarType
from hidet.ir.expr import Expr, Var, var
from hidet.ir.layout import TaskLayout, TaskLayoutExpander

ScopedStmt = Union[IfStmt, LetStmt, ForStmt]


class StmtScope:
    def __init__(self, sb: 'StmtBuilder', stmts: Union[Sequence[ScopedStmt], ScopedStmt], ret=None):
        if isinstance(stmts, Stmt):
            stmts = [stmts]
        self.sb = sb
        self.stmts = stmts
        self.ret = ret

    def __enter__(self):
        for stmt in self.stmts:
            self.sb.enter_body(stmt)
        return self.ret

    def __exit__(self, exc_type, exc_val, exc_tb):
        for _ in self.stmts:
            self.sb.exit_body()


class StmtBuilder:
    def __init__(self):
        self.scope_stack = [[]]

    def __iadd__(self, other: Union[Stmt, Expr]):
        assert isinstance(other, (Stmt, Expr))
        self.append(other)
        return self

    def let(self, v: Union[str, Var], value: Expr) -> StmtScope:
        if isinstance(v, str):
            v = var(v)
        return StmtScope(self, stmts=LetStmt(v, value), ret=v)

    def for_loop(self, v: Union[str, Var], extent: Expr) -> StmtScope:
        if isinstance(v, str):
            v = var(v)
        return StmtScope(self, stmts=ForStmt(v, extent), ret=v)

    def if_then(self, cond: Expr) -> StmtScope:
        return StmtScope(self, stmts=[IfStmt(cond)], ret=None)

    def otherwise(self) -> StmtScope:
        assert len(self.scope_stack[-1]) > 0
        if_stmt = self.scope_stack[-1].pop()
        assert isinstance(if_stmt, IfStmt)
        assert if_stmt.then_body is not None
        assert if_stmt.else_body is None
        return StmtScope(self, stmts=if_stmt, ret=None)

    def for_task_fields(self, worker_index: Expr, task_layout: TaskLayout) -> List[Sequence[Expr]]:
        expander = TaskLayoutExpander()
        fields = expander.expand(worker_index, task_layout)
        return StmtScope(self, stmts=expander.stmts, ret=fields)

    def append(self, stmt: Union[Stmt, Expr]):
        if stmt is None:
            return
        if not isinstance(stmt, Stmt):
            assert isinstance(stmt, Expr)
            stmt = EvaluateStmt(stmt)
        self.scope_stack[-1].append(stmt)

    def enter_body(self, stmt: Union[LetStmt, IfStmt, ForStmt]):
        self.scope_stack[-1].append(stmt)
        self.scope_stack.append([])

    def exit_body(self):
        body = SeqStmt(self.scope_stack.pop())
        assert len(self.scope_stack) > 0
        last_stmt = self.scope_stack[-1][-1]
        if isinstance(last_stmt, (LetStmt, ForStmt)):
            assert last_stmt.body is None
            last_stmt.body = body
        elif isinstance(last_stmt, IfStmt):
            if last_stmt.then_body is None:
                last_stmt.then_body = body
            else:
                assert last_stmt.else_body is None
                last_stmt.else_body = body
        else:
            assert False

    def finish(self):
        assert len(self.scope_stack) == 1
        return SeqStmt(self.scope_stack[0])
