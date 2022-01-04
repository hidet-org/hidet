from typing import List
from copy import copy
from hidet.ir.expr import Axis


class Stmt:
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
    def __init__(self, loop_var, body=None):
        super().__init__()
        self.loop_var: Axis = loop_var
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
                assert stmt.then_body is None
                nstmt = stmt.copy()
                nstmt.then_body = body
                body = nstmt
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


