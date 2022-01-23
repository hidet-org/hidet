from typing import Union, Optional

from hidet.ir.stmt import Stmt, ForStmt, IfStmt, LetStmt, EvaluateStmt, SeqStmt
from hidet.ir.type import TypeNode, scalar_type, ScalarType
from hidet.ir.expr import Expr, Var, var


class StmtScope:
    def __init__(self, sb: 'StmtBuilder', ret=None):
        self.sb = sb
        self.ret = ret

    def __enter__(self):
        self.sb.enter_body()
        return self.ret

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.sb.exit_body()


class StmtBuilder:
    def __init__(self):
        self.scope_stack = [[]]

    def __iadd__(self, other: Union[Stmt, Expr]):
        assert isinstance(other, (Stmt, Expr))
        self.append(other)
        return self

    def let(self, v: Union[str, Var], value) -> StmtScope:
        if isinstance(v, str):
            v = var(v)
        self.append(LetStmt(v, value))
        return StmtScope(self, v)

    def for_loop(self, v: Union[str, Var], extent: Expr) -> StmtScope:
        if isinstance(v, str):
            v = var(v)
        self.append(ForStmt(v, extent))
        return StmtScope(self, v)

    def if_then(self, cond: Expr) -> StmtScope:
        self.append(IfStmt(cond))
        return StmtScope(self)

    def otherwise(self) -> StmtScope:
        assert len(self.scope_stack[-1]) > 0
        if_stmt = self.scope_stack[-1][-1]
        assert isinstance(if_stmt, IfStmt)
        assert if_stmt.then_body is not None
        assert if_stmt.else_body is None
        return StmtScope(self)

    def append(self, stmt: Union[Stmt, Expr]):
        if stmt is None:
            return
        if not isinstance(stmt, Stmt):
            assert isinstance(stmt, Expr)
            stmt = EvaluateStmt(stmt)
        self.scope_stack[-1].append(stmt)

    def enter_body(self):
        assert len(self.scope_stack[-1]) > 0
        last_stmt = self.scope_stack[-1][-1]
        assert isinstance(last_stmt, (IfStmt, ForStmt, LetStmt))
        self.scope_stack.append([])

    def exit_body(self, num_scopes=1):
        while num_scopes >= 1:
            num_scopes -= 1
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
                    last_stmt.else_body = None
            else:
                assert False

    def finish(self):
        assert len(self.scope_stack) == 1
        return SeqStmt(self.scope_stack[0])
