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
from hidet.ir import Var, ForMappingStmt, Stmt, ForStmt, Expr, SeqStmt, IRModule

from hidet.ir.dtypes import int32
from hidet.ir.expr import var
from hidet.ir.mapping import TaskMapping, SpatialTaskMapping, RepeatTaskMapping, ComposedTaskMapping
from hidet.ir.func import Function
from hidet.ir.functors import IRRewriter
from hidet.ir.tools import rewrite, simplify
from hidet.ir.tools.rewriter import PolinomialExpr2ExprRewriter
from hidet.transforms.base import Pass
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
        if isinstance(mapping, SpatialTaskMapping) and len(loop_vars) != 0:
            strides = strides_from_ranks(shape=mapping.task_shape, ranks=mapping.ranks)
            flatten = int32.zero
            for loop_var, stride in zip(loop_vars, strides):
                flatten += loop_var * stride
            rewriter = PolinomialExpr2ExprRewriter(flatten, worker)
            body = rewriter.rewrite(body)

        tasks = self.visit(mapping, worker)
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
        if len(strides) != 0:
            for i, (extent, stride) in enumerate(zip(mapping.task_shape, strides)):
                # For first index we don't need to do `% extent` because `taks_size == num_workers`
                # It's guarantee by `task_mapping_bound_check_pass()`
                if mapping.ranks[i] == 0:
                    task.append(worker // stride)
                else:
                    task.append(worker // stride % extent)
        return [task]

    def visit_Repeat(self, mapping: RepeatTaskMapping, worker: Expr) -> List[TaskIndex]:
        # pylint: disable=unused-argument
        # worker is unused because there is only a single worker with index 0
        num_loops = len(mapping.task_shape)
        task: List[Optional[Var]] = [None for _ in range(num_loops)]
        for i in range(num_loops):
            dim = mapping.ranks.index(i)
            extent = simplify(mapping.task_shape[dim])
            attr = mapping.attrs[dim]
            loop_var = var('i')
            self.loop_nests.append(ForStmt(loop_var=loop_var, extent=extent, attr=attr))
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


class LowerTaskMappingRewriter(IRRewriter):
    def visit_ForTaskStmt(self, stmt: ForMappingStmt):
        new_body = self.visit(stmt.body)
        expander = TaskMappingExpander()
        new_stmt = expander.expand(mapping=stmt.mapping, worker=stmt.worker, loop_vars=stmt.loop_vars, body=new_body)
        return new_stmt


class LowerTaskMappingPass(Pass):
    def __init__(self, name=None):
        self.block_dim = None
        super().__init__(name)

    def process_module(self, ir_module: IRModule) -> IRModule:
        for func in ir_module.functions.values():
            if func.kind == 'cuda_kernel':
                self.block_dim = func.attrs['cuda.block_dim']
        return super().process_module(ir_module)

    def process_func(self, func: Function) -> Function:
        rewriter = LowerTaskMappingRewriter()
        return rewriter.rewrite(func)


def lower_task_mapping_pass() -> Pass:
    return LowerTaskMappingPass()
