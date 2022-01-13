from typing import List
from copy import copy
from hidet.ir.node import Node
from hidet.ir.expr import Var


class Stmt(Node):
    def copy(self):
        return copy(self)


class EvaluateStmt(Stmt):
    def __init__(self, expr):
        super().__init__()
        self.expr = expr


class BufferStoreStmt(Stmt):
    def __init__(self, buf, indices, value):
        super().__init__()
        self.buf = buf
        self.indices = indices
        self.value = value


class AssignStmt(Stmt):
    def __init__(self, var, value):
        super().__init__()
        self.var = var
        self.value = value


class LetStmt(Stmt):
    def __init__(self, var, value, body=None):
        super().__init__()
        self.var = var
        self.value = value
        self.body = body


class ForStmt(Stmt):
    def __init__(self, loop_var, extent, body=None):
        super().__init__()
        self.loop_var: Var = loop_var
        self.extent = extent
        self.body = body


class IfStmt(Stmt):
    def __init__(self, cond, then_body=None, else_body=None):
        super().__init__()
        self.cond = cond
        self.then_body = then_body
        self.else_body = else_body


class AssertStmt(Stmt):
    def __init__(self, cond, msg):
        super().__init__()
        self.cond = cond
        self.msg = msg


class SeqStmt(Stmt):
    def __init__(self, seq):
        super().__init__()
        self.seq: List = seq
        for stmt in seq:
            assert isinstance(stmt, Stmt)

    def append(self, stmt):
        self.seq.append(stmt)
        assert isinstance(stmt, Stmt)


def flatten(stmts):
    flattened = []
    for stmt in stmts:
        if isinstance(stmt, SeqStmt):
            flattened.extend(flatten(stmt.seq))
        else:
            flattened.append(stmt)
    return flattened


def concat_stmts(stmts):
    # stmts = flatten(stmts)
    body = None
    for stmt in reversed(stmts):
        if body is None:
            body = stmt
            if isinstance(stmt, IfStmt):
                assert stmt.then_body is not None
            if isinstance(stmt, LetStmt):
                assert stmt.body is not None
            if isinstance(stmt, ForStmt):
                assert stmt.body is not None
        else:
            if isinstance(stmt, IfStmt):
                if stmt.then_body is None:
                    nstmt = stmt.copy()
                    nstmt.then_body = body
                    body = nstmt
                elif stmt.else_body is None:
                    nstmt = stmt.copy()
                    nstmt.else_body = body
                    body = nstmt
                else:
                    raise ValueError()
            elif isinstance(stmt, LetStmt):
                assert stmt.body is None
                nstmt = stmt.copy()
                nstmt.body = body
                body = nstmt
            elif isinstance(stmt, ForStmt):
                assert stmt.body is None
                nstmt = stmt.copy()
                nstmt.body = body
                body = nstmt
            else:
                raise ValueError()
    return body


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

    def append(self, stmt: Stmt):
        if stmt is None:
            return
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
