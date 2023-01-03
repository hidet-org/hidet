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
import itertools
from hidet.ir import Var, ForTaskStmt, Stmt, ForStmt, Expr, SeqStmt
from hidet.ir.expr import var
from hidet.ir.mapping import TaskMapping, SpatialTaskMapping, RepeatTaskMapping, ComposedTaskMapping
from hidet.transforms.base import Pass, FunctionBodyPass
from hidet.ir.functors import StmtExprRewriter, rewrite, simplify
from hidet.utils import prod

Int = Union[Expr, int]
TaskIndex = List[Int]


def strides_from_ranks(shape: Sequence[int], ranks: Sequence[int]) -> List[int]:
    assert len(shape) == len(ranks) == len(set(ranks)) and all(0 <= v < len(ranks) for v in ranks)
    strides = []
    for i in range(len(shape)):
        strides.append(prod([extent for extent, rank in zip(shape, ranks) if rank > ranks[i]]))
    return strides


class TaskMappingExpander:
    def __init__(self):
        self.loop_nests: List[ForStmt] = []

    def expand(self, mapping: TaskMapping, worker: Expr, loop_vars: List[Var], body: Stmt) -> Stmt:
        tasks: List[TaskIndex] = self.visit(mapping, worker)
        seq = []
        for task in tasks:
            remap: Dict[Var, Expr] = {a: b for a, b in zip(loop_vars, task)}
            seq.append(rewrite(body, remap))
        body = SeqStmt(seq)
        for loop in reversed(self.loop_nests):
            loop.body = body
            body = loop
        return body

    def visit(self, mapping: TaskMapping, worker: Expr) -> List[TaskIndex]:
        if not isinstance(mapping, TaskMapping):
            raise ValueError('Expect a task mapping, got {}'.format(type(mapping).__name__))
        if isinstance(mapping, SpatialTaskMapping):
            return self.visit_Spatial(mapping, worker)
        elif isinstance(mapping, RepeatTaskMapping):
            return self.visit_Repeat(mapping, worker)
        elif isinstance(mapping, ComposedTaskMapping):
            return self.visit_Composed(mapping, worker)
        else:
            raise NotImplementedError()

    def visit_Spatial(self, mapping: SpatialTaskMapping, worker: Expr) -> List[TaskIndex]:
        strides = strides_from_ranks(shape=mapping.task_shape, ranks=mapping.ranks)
        task = []
        for extent, stride in zip(mapping.task_shape, strides):
            task.append(worker // stride % extent)
        return [task]

    def visit_Repeat(self, mapping: RepeatTaskMapping, worker: Expr) -> List[TaskIndex]:
        # pylint: disable=unused-argument
        # worker is unused because there is only a single worker with index 0
        unroll = False
        if unroll:

            def global_index(task: Sequence[int], strides: Sequence[int]) -> int:
                return sum(a * b for a, b in zip(task, strides))

            strides = strides_from_ranks(shape=mapping.task_shape, ranks=mapping.ranks)
            ranges = [range(s) for s in mapping.task_shape]
            tasks = list(tuple(task) for task in itertools.product(*ranges))
            tasks = sorted(tasks, key=lambda task: global_index(task, strides))
            return [list(task) for task in tasks]
        else:
            num_loops = len(mapping.task_shape)
            task: List[Optional[Var]] = [None for _ in range(num_loops)]
            for i in range(num_loops):
                dim = mapping.ranks.index(i)
                extent = simplify(mapping.task_shape[dim])
                loop_var = var('i')
                self.loop_nests.append(ForStmt(loop_var=loop_var, extent=extent))
                task[dim] = loop_var
            return [task]

    def visit_Composed(self, mapping: ComposedTaskMapping, worker: Expr) -> List[TaskIndex]:
        outer, inner = mapping.outer, mapping.inner
        outer_worker, inner_worker = worker // inner.num_workers, worker % inner.num_workers
        outer_tasks = self.visit(outer, outer_worker)
        inner_tasks = self.visit(inner, inner_worker)
        tasks = []
        for outer_task in outer_tasks:
            for inner_task in inner_tasks:
                task = [a * inner.task_shape[i] + b for i, (a, b) in enumerate(zip(outer_task, inner_task))]
                tasks.append(task)
        return tasks


class LowerTaskMappingRewriter(StmtExprRewriter):
    def visit_ForTaskStmt(self, stmt: ForTaskStmt):
        body = self.visit(stmt.body)
        expander = TaskMappingExpander()
        return expander.expand(mapping=stmt.mapping, worker=stmt.worker, loop_vars=stmt.loop_vars, body=body)


class LowerTaskMappingPass(FunctionBodyPass):
    def process_body(self, stmt: Stmt) -> Stmt:
        rewriter = LowerTaskMappingRewriter()
        return rewriter.rewrite(stmt)


def lower_task_mapping_pass() -> Pass:
    return LowerTaskMappingPass()
