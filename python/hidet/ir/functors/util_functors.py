# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Union, Mapping
from hidet.ir.expr import Let, Var, Expr
from hidet.ir.func import Function
from hidet.ir.stmt import Stmt, ForStmt, LetStmt

from .base import StmtExprVisitor, StmtExprRewriter, FuncStmtExprVisitor


class StmtExprMapRewriter(StmtExprRewriter):
    def __init__(self, rmap):
        super().__init__()
        self.rmap = rmap

    def visit(self, node):
        if node not in self.memo:
            if node in self.rmap:
                self.memo[node] = self.rmap[node]
            else:
                self.memo[node] = StmtExprRewriter.visit(self, node)
        return self.memo[node]


class SubStmtExprCollector(FuncStmtExprVisitor):
    def __init__(self, expr_types, stop_when_found=False):
        super().__init__()
        self.expr_types = expr_types
        self.stop_when_found = stop_when_found
        self.exprs = []

    def collect(self, e):
        self.exprs.clear()
        self.visit(e)
        return self.exprs

    def visit(self, node):
        if node in self.memo:
            return self.memo[node]
        if isinstance(node, self.expr_types):
            self.exprs.append(node)
            if self.stop_when_found:
                self.memo[node] = None
                return None
        return StmtExprVisitor.visit(self, node)


class FreeVarCollector(StmtExprVisitor):
    def __init__(self):
        super().__init__()
        self.defined = set()
        self.free_vars = set()

    def collect(self, e):
        self.defined.clear()
        self.visit(e)
        return self.free_vars

    def visit_LetStmt(self, stmt: LetStmt):
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

    def visit_LetStmt(self, stmt: LetStmt):
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


def rewrite(node: Union[Expr, Stmt, tuple], rewrite_map: Mapping[Union[Stmt, Expr], Union[Stmt, Expr]]):
    assert isinstance(rewrite_map, dict)
    rewriter = StmtExprMapRewriter(rewrite_map)
    return rewriter.rewrite(node)


def collect(node: Union[Function, Expr, Stmt, list, tuple], node_types, stop_when_found=False) -> list:
    """
    Collect sub-nodes in given node with specific types.

    Parameters
    ----------
    node: Union[Function, Expr, Stmt, list, tuple]
        The root node to start collecting.
    node_types: Sequence[Type[Union[Stmt, Expr]]], or Type[Stmt], or Type[Expr]
        The node types to collect, can be arbitrary subclass of Expr and Stmt
    stop_when_found: bool
        When found node of given type, whether to collect the sub-nodes of that node.

    Returns
    -------
    ret: List[Any]
        The collected nodes.

    """
    if not isinstance(node_types, tuple):
        if isinstance(node_types, list):
            node_types = tuple(node_types)
        elif issubclass(node_types, (Stmt, Expr)):
            node_types = (node_types,)
        else:
            raise ValueError()
    if isinstance(node, list):
        node = tuple(node)

    collector = SubStmtExprCollector(node_types, stop_when_found)
    collected = collector.collect(node)
    return collected


def clone(node: Union[Stmt, Expr]) -> Union[Stmt, Expr]:
    return CloneRewriter()(node)


def collect_free_vars(node: Union[Expr, Stmt]):
    collector = FreeVarCollector()
    return collector.collect(node)
