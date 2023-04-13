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
from typing import Union, Sequence
from hidet.ir.expr import Var, Expr
from hidet.ir.compute import GridCompute, ReduceCompute, ArgReduceCompute
from hidet.ir.stmt import ForStmt, LetStmt, SeqStmt, DeclareStmt, Stmt
from hidet.ir.functors import IRVisitor


class FreeVarCollector(IRVisitor):
    def __init__(self):
        super().__init__()
        self.defined = set()
        self.free_vars = set()

    def collect(self, e):
        self.defined.clear()
        self.visit(e)
        return self.free_vars

    def visit_GridCompute(self, node: GridCompute):
        for v in node.axes:
            self.defined.add(v)
        self.visit(node.value)
        for v in node.axes:
            self.defined.remove(v)

    def visit_ReduceCompute(self, node: ReduceCompute):
        for v in node.axes:
            self.defined.add(v)
        self.visit(node.value)
        for v in node.axes:
            self.defined.remove(v)

    def visit_ArgReduceCompute(self, node: ArgReduceCompute):
        self.defined.add(node.axis)
        self.visit(node.value)
        self.defined.remove(node.axis)

    def visit_LetStmt(self, stmt: LetStmt):
        for bind_var, bind_value in zip(stmt.bind_vars, stmt.bind_values):
            self.visit(bind_value)
            self.defined.add(bind_var)
        self.visit(stmt.body)
        for bind_var in stmt.bind_vars:
            self.defined.remove(bind_var)

    def visit_ForStmt(self, stmt: ForStmt):
        self.defined.add(stmt.loop_var)
        super().visit_ForStmt(stmt)
        self.defined.remove(stmt.loop_var)

    def visit_SeqStmt(self, stmt: SeqStmt):
        added = []
        for s in stmt.seq:
            if isinstance(s, DeclareStmt):
                self.defined.add(s.var)
                added.append(s.var)
            self.visit(s)
        for v in added:
            self.defined.remove(v)

    def visit_Var(self, e: Var):
        if e not in self.defined:
            self.free_vars.add(e)


def collect_free_vars(node: Union[Expr, Stmt, Sequence[Union[Expr, Stmt]]]):
    if not isinstance(node, (list, tuple)):
        node = [node]
    collector = FreeVarCollector()
    for n in node:
        collector.collect(n)
    return collector.free_vars
