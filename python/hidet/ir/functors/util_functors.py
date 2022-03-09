from typing import Union, Mapping
from hidet.ir.expr import Let
from hidet.ir.func import Function
from hidet.ir.stmt import Stmt, ForStmt, LetStmt
from hidet.ir.dialects.compute import *

from .base import StmtExprVisitor, StmtExprRewriter, FuncStmtExprVisitor


class StmtExprMapRewriter(StmtExprRewriter):
    def __init__(self, rmap):
        super().__init__()
        self.rmap = rmap

    def visit(self, e):
        if e not in self.memo:
            if e in self.rmap:
                self.memo[e] = self.rmap[e]
            else:
                self.memo[e] = StmtExprRewriter.visit(self, e)
        return self.memo[e]


class SubStmtExprCollector(FuncStmtExprVisitor):
    def __init__(self, expr_types):
        super().__init__()
        self.expr_types = expr_types
        self.exprs = []

    def collect(self, e):
        self.exprs.clear()
        self.visit(e)
        return self.exprs

    def visit(self, e):
        if e in self.memo:
            return self.memo[e]
        if isinstance(e, self.expr_types):
            self.exprs.append(e)
        StmtExprVisitor.visit(self, e)


class FreeVarCollector(StmtExprVisitor):
    def __init__(self):
        super().__init__()
        self.defined = set()
        self.free_vars = set()

    def collect(self, e):
        self.defined.clear()
        self.visit(e)
        return self.free_vars

    def visit_SeqLetStmt(self, stmt: LetStmt):
        for bind_var, bind_value in zip(stmt.bind_vars, stmt.bind_values):
            self.visit(bind_value)
            self.defined.add(bind_var)
        self.visit(stmt.body)
        for bind_var in stmt.bind_vars:
            self.defined.remove(bind_var)

    def visit_ForStmt(self, stmt: ForStmt):
        self.defined.add(stmt.loop_var)
        StmtExprVisitor.visit_ForStmt(self, stmt)
        self.defined.remove(stmt.loop_var)

    def visit_Var(self, e: Var):
        if e not in self.defined:
            self.free_vars.add(e)


class CloneRewriter(StmtExprRewriter):
    def clone(self, obj: Union[Stmt, Expr]):
        return self(obj)

    def visit_SeqLetStmt(self, stmt: LetStmt):
        bind_vars = []
        bind_values = []
        for bind_var, bind_value in zip(stmt.bind_vars, stmt.bind_values):
            bind_vars.append(Var(bind_var.hint, bind_var.type))
            self.memo[bind_var] = bind_vars[-1]
            bind_values.append(self(bind_value))
        return LetStmt(bind_vars, bind_values, self(stmt.body))

    def visit_Let(self, e: Let):
        v = Var(e.var.hint, e.var.type)
        self.memo[e.var] = v
        return Let(v, self(e.value), self(e.body))


def rewrite(node: Union[Expr, Stmt], rewrite_map: Mapping[Expr, Expr]):
    assert isinstance(rewrite_map, dict)
    rewriter = StmtExprMapRewriter(rewrite_map)
    return rewriter.rewrite(node)


def collect(node: Union[Function, Expr, Stmt], node_types) -> list:
    if not isinstance(node_types, tuple):
        if isinstance(node_types, list):
            node_types = tuple(node_types)
        elif issubclass(node_types, (Stmt, Expr)):
            node_types = (node_types,)
        else:
            raise ValueError()

    collector = SubStmtExprCollector(node_types)
    return collector.collect(node)


def clone(node: Union[Stmt, Expr]) -> Union[Stmt, Expr]:
    return CloneRewriter()(node)


def collect_free_vars(node: Union[Expr, Stmt]):
    collector = FreeVarCollector()
    return collector.collect(node)
