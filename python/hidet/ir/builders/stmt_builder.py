from typing import Union

from hidet.ir.stmt import Stmt, ForStmt, IfStmt, LetStmt, EvaluateStmt, SeqStmt
from hidet.ir.expr import Expr


class StmtBuilder:
    def __init__(self):
        self.scope_stack = [[]]

    def __enter__(self):
        self.enter_body()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.exit_body()

    def for_body(self):
        assert len(self.scope_stack[-1]) > 0
        for_stmt = self.scope_stack[-1][-1]
        assert isinstance(for_stmt, ForStmt)
        assert for_stmt.body is None
        return self

    def then_body(self):
        assert len(self.scope_stack[-1]) > 0
        if_stmt = self.scope_stack[-1][-1]
        assert isinstance(if_stmt, IfStmt)
        assert if_stmt.then_body is None
        return self

    def else_body(self):
        assert len(self.scope_stack[-1]) > 0
        if_stmt = self.scope_stack[-1][-1]
        assert isinstance(if_stmt, IfStmt)
        assert if_stmt.then_body is None
        assert if_stmt.else_body is None
        return self

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
                last_stmt.else_body = None
        else:
            assert False

    def finish(self):
        assert len(self.scope_stack) == 1
        return SeqStmt(self.scope_stack[0])
