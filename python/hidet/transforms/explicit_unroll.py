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
from typing import List, Union

from hidet.ir.dtypes import int32
from hidet.ir.expr import Constant
from hidet.ir.functors import IRRewriter
from hidet.ir.stmt import Stmt, ForStmt, Expr, SeqStmt
from hidet.ir.tools import simplify, rewrite
from hidet.transforms.base import Pass, FunctionBodyPass

Int = Union[Expr, int]
TaskIndex = List[Int]


class ExplicitUnrollRewriter(IRRewriter):
    def visit_ForStmt(self, stmt: ForStmt):
        if stmt.attr.unroll and stmt.attr.unroll_explicit:
            if not isinstance(stmt.attr.unroll, bool):
                raise NotImplementedError('Explicit unroll with unroll factor is not supported yet')
            extent_expr: Expr = simplify(stmt.extent)
            if not isinstance(extent_expr, Constant):
                raise ValueError('Expect a constant extent to unroll explicitly')
            else:
                extent_int = int(extent_expr)

            body = self.visit(stmt.body)

            seq: List[Stmt] = []
            for i in range(extent_int):
                seq.append(rewrite(body, {stmt.loop_var: int32(i)}, clone_internal_var=True))
            if len(seq) == 1:
                return seq[0]
            else:
                return SeqStmt(seq)
        return IRRewriter.visit_ForStmt(self, stmt)


class ExplicitUnrollPass(FunctionBodyPass):
    def process_body(self, stmt: Stmt) -> Stmt:
        rewriter = ExplicitUnrollRewriter()
        return rewriter.rewrite(stmt)


def explicit_unroll_pass() -> Pass:
    return ExplicitUnrollPass()
