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
from typing import Dict, List, Union, Mapping

from hidet.ir.expr import Let, Var
from hidet.ir.functors import IRRewriter
from hidet.ir.node import Node
from hidet.ir.stmt import ForMappingStmt, DeclareStmt, ForStmt
from hidet.ir.stmt import LetStmt


class MapBasedRewriter(IRRewriter):
    def __init__(self, rmap):
        super().__init__()
        self.memo.update(rmap)


class CloneRewriter(IRRewriter):
    """
    A rewriter that will create a new var for each statement/expr that will declare vars
    """

    def __init__(self, remap: Dict[Node, Node]):
        super().__init__()
        self.memo.update(remap)

    def process_var(self, v: Var):
        visited_v = self.visit(v)
        if visited_v is v:
            new_var = Var(v.hint, type=v.type, name=v.name)
        else:
            new_var = visited_v
        self.memo[v] = new_var
        return new_var

    def visit_ForStmt(self, stmt: ForStmt):
        loop_var = self.process_var(stmt.loop_var)
        extent = self.visit(stmt.extent)
        body = self.visit(stmt.body)
        return ForStmt(loop_var, extent, body, attr=stmt.attr)

    def visit_ForTaskStmt(self, stmt: ForMappingStmt):
        loop_vars: List[Var] = [self.process_var(v) for v in stmt.loop_vars]
        worker = self.visit(stmt.worker)
        body = self.visit(stmt.body)
        return ForMappingStmt(loop_vars=loop_vars, mapping=stmt.mapping, worker=worker, body=body)

    def visit_LetStmt(self, stmt: LetStmt):
        bind_vars = [self.process_var(v) for v in stmt.bind_vars]
        bind_values = [self.visit(bind_value) for bind_value in stmt.bind_values]
        body = self.visit(stmt.body)
        return LetStmt(bind_vars, bind_values, body)

    def visit_DeclareStmt(self, stmt: DeclareStmt):
        v = self.process_var(stmt.var)
        init = self.visit(stmt.init) if stmt.init is not None else None
        return DeclareStmt(v, init, stmt.is_static)

    def visit_Let(self, e: Let):
        v = self.process_var(e.var)
        value = self.visit(e.value)
        body = self.visit(e.body)
        return Let(v, value, body)


def rewrite(node: Union[Node, tuple, list, dict], rewrite_map: Mapping[Node, Node], clone_internal_var=False):
    assert isinstance(rewrite_map, dict)
    if clone_internal_var:
        rewriter = CloneRewriter(rewrite_map)
    else:
        rewriter = MapBasedRewriter(rewrite_map)
    return rewriter.rewrite(node)
