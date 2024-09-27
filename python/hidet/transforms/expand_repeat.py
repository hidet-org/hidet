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
from typing import List, Dict, Sequence, Union, Optional
from hidet.ir import Var, ForMappingStmt, Stmt, ForStmt, Expr
from hidet.ir.expr import var
from hidet.ir.mapping import RepeatTaskMapping, ComposedTaskMapping
from hidet.ir.func import Function
from hidet.ir.functors import IRRewriter
from hidet.ir.tools import rewrite, simplify
from hidet.transforms.base import Pass
from hidet.utils import prod
from hidet.ir.mapping import mapping_2_list, list_2_mapping


Int = Union[Expr, int]
TaskIndex = List[Int]


# Expand one leading repeat task mapping.
#
# repeat(a).anothertaskmapping(b).anothertaskmapping(b). ...
# to
# for ...
#     anothertaskmapping(b).anothertaskmapping(b). ...
#
# This pass simplify task mapping that allow to apply
# more efficient task mapping expand on final expand of task mapping
class ExpandRepeatRewriter(IRRewriter):
    def __init__(self):
        super().__init__()
        self.loop_nests: List[ForStmt] = []

    def visit_ForTaskStmt(self, stmt: ForMappingStmt):
        new_body = self.visit(stmt.body)
        if new_body is stmt.body:
            new_stmt = stmt
        else:
            new_stmt = ForMappingStmt(loop_vars=stmt.loop_vars, mapping=stmt.mapping, worker=stmt.worker, body=new_body)

        new_stmt2 = self._visit(new_stmt)
        # ExpandRepeatRewriter expand one leading repeat() only. Run in loop to expand all leading repeat()
        while new_stmt2 is not new_stmt:
            new_stmt = new_stmt2
            new_stmt2 = self.visit(new_stmt)

        return new_stmt2

    def _visit(self, stmt: ForMappingStmt):
        self.loop_nests: List[ForStmt] = []
        new_stmt = stmt
        if isinstance(stmt.mapping, ComposedTaskMapping):
            task = self._visit_Composed(stmt.mapping, stmt.worker, stmt.body, stmt.loop_vars)
            if task is not None:
                remap: Dict[Var, Expr] = {a: b for a, b in zip(stmt.loop_vars, task)}
                new_body = rewrite(stmt.body, remap)
                for loop in reversed(self.loop_nests):
                    loop.body = new_body
                    new_body = loop
                new_stmt = new_body

        return new_stmt

    def _visit_Composed(
        self, mapping: ComposedTaskMapping, worker: Expr, body: Stmt, loop_vars: Sequence[Var]
    ) -> List[TaskIndex]:
        mappings = mapping_2_list(mapping)

        mapp = mappings[-1]
        if not isinstance(mapp, RepeatTaskMapping):
            return None

        num_loops = len(mapp.task_shape)
        task: List[Optional[Var]] = [None for _ in range(num_loops)]

        for i in range(num_loops):
            dim = mapp.ranks.index(i)
            extent = simplify(mapp.task_shape[dim])
            attr = mapp.attrs[dim]

            loop_var = var('r')
            self.loop_nests.append(ForStmt(loop_var=loop_var, extent=extent, attr=attr))
            task[dim] = loop_var

        mappings = mappings[0:-1]
        num_vars = len(loop_vars)
        tmp = [[mapp.task_shape[i] for mapp in mappings] for i in range(num_vars)]
        mul = [prod(tmp[i]) for i in range(num_vars)]
        res_task = [a * mul[i] + b for i, (a, b) in enumerate(zip(task, loop_vars))]

        self.loop_nests.append(
            ForMappingStmt(loop_vars=loop_vars, mapping=list_2_mapping(mappings), worker=worker, body=body)
        )
        return res_task


class ExpandRepeatPass(Pass):
    def process_func(self, func: Function) -> Function:
        rewriter = ExpandRepeatRewriter()
        return rewriter.rewrite(func)


def expand_repeat_mapping_pass() -> Pass:
    return ExpandRepeatPass()
