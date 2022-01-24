from typing import List, Union, Optional
from copy import copy
from hidet.ir.node import Node
from hidet.ir.expr import Var, Expr, convert, Constant


class Stmt(Node):
    def copy(self):
        return copy(self)


class EvaluateStmt(Stmt):
    def __init__(self, expr):
        super().__init__()
        self.expr = convert(expr)


class BufferStoreStmt(Stmt):
    def __init__(self, buf, indices, value):
        super().__init__()
        self.buf = buf
        self.indices = convert(indices)
        self.value = convert(value)


class AssignStmt(Stmt):
    def __init__(self, var, value):
        super().__init__()
        self.var = var
        self.value = convert(value)


class LetStmt(Stmt):
    def __init__(self, var, value, body=None):
        super().__init__()
        self.var = var
        self.value = convert(value)
        self.body = body


class ForStmt(Stmt):
    DEFAULT_UNROLL_LIMIT = 32

    def __init__(self, loop_var, extent, unroll: Optional[bool] = None, body=None):
        from hidet.ir.functors import simplify
        super().__init__()
        self.loop_var: Var = loop_var
        self.extent = simplify(convert(extent))
        if unroll is None:
            if isinstance(self.extent, Constant) and self.extent.value <= ForStmt.DEFAULT_UNROLL_LIMIT:
                self.unroll = True
            else:
                self.unroll = None  # leave to the underlying compiler to determine the unroll strategy
        else:
            self.unroll = unroll
        self.body = body


class IfStmt(Stmt):
    def __init__(self, cond, then_body=None, else_body=None):
        super().__init__()
        self.cond = convert(cond)
        self.then_body = then_body
        self.else_body = else_body


class AssertStmt(Stmt):
    def __init__(self, cond, msg):
        super().__init__()
        self.cond = convert(cond)
        self.msg = msg


class BlackBoxStmt(Stmt):
    def __init__(self, stmt_str: str):
        super().__init__()
        self.stmt_str = stmt_str


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
