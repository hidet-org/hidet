from typing import Union
from hidet.ir.stmt import Stmt
from hidet.ir.dialects.compute import *

from .base import StmtExprVisitor, StmtExprRewriter


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


class SubStmtExprCollector(StmtExprVisitor):
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


def rewrite(node: Union[Expr, Stmt], rewrite_map):
    rewriter = StmtExprMapRewriter(rewrite_map)
    return rewriter.rewrite(node)


def collect(node: Union[Expr, Stmt], node_types):
    if not isinstance(node_types, tuple):
        if isinstance(node_types, list):
            node_types = tuple(node_types)
        elif issubclass(node_types, (Stmt, Expr)):
            node_types = (node_types,)
        else:
            raise ValueError()

    collector = SubStmtExprCollector(node_types)
    return collector.collect(node)

