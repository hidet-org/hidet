from typing import Mapping, List, Tuple
from functools import reduce
import operator

from hidet.ir.node import Node
from hidet.ir.type import scalar_type
from hidet.ir.expr import Expr, Var, Call, convert, And, TensorElement, var
from hidet.ir.stmt import IfStmt, EvaluateStmt, concat_stmts, SeqStmt, BufferStoreStmt, Stmt, LetStmt
from hidet.ir.task import Task, Grid, Thread
from hidet.ir.func import IRModule, Function
from hidet.ir.dialects.compute import TensorInput, ScalarInput, TensorCompute, scalar_input
from hidet.ir.dialects.pattern import TaskPattern, TensorComputePattern
from hidet.ir.dialects.lowlevel import VoidType, ReferenceType
from hidet.ir.functors import rewrite, infer_type, collect, simplify
from hidet.implement.implementer import Implementer, implement, register_impl
from hidet.ir.primitives import thread_idx, block_idx


def reduce_product(lst: List[Expr]):
    if len(lst) == 0:
        return convert(1)
    s = lst[0]
    for v in lst[1:]:
        s = s * v
    return s


@register_impl('cuda_grid_naive_implementer')
class CudaGridNaiveImplementer(Implementer):
    """
    Naive grid implementor that implements a task with TensorCompute. It map each element in the compute grid to a thread.
    This implementor only accepts task with static compute shape.
    """

    def __init__(self):
        self.pattern = TaskPattern(
            compute_pattern=TensorComputePattern(rank=None, allow_dynamic_axis=False),
            required_params=[],
            required_param_types=[],
            allow_tensor_extra_params=True,
            worker=Grid(grid_dim=None, block_dim=None)
        )

    def priority(self) -> int:
        return 0

    def task_pattern(self) -> TaskPattern:
        return self.pattern

    @staticmethod
    def grid_thread_map(grid_shape: List[Expr],
                        block_shape: List[Expr],
                        task_shape: List[Expr]
                        ) -> Tuple[List[Stmt], Expr, List[Expr], Grid]:
        """
        Map the block idx and thread idx to the indices in the shape.
        Returns stmt, cond, indices
        Return the stmt to calculate the mapping and the indices where cond protect the indices in the shape
        """
        block_index: Expr = block_idx()
        thread_index: Expr = thread_idx()
        rank = len(task_shape)
        assert len(grid_shape) == rank and len(block_shape) == rank
        task_indices = []
        stmts = []
        cond = convert(True)
        for i in range(rank):
            block_idx_value = block_index / reduce_product(grid_shape[i + 1:]) % grid_shape[i]
            thread_idx_value = thread_index / reduce_product(block_shape[i + 1:]) % block_shape[i]
            task_idx_value = block_idx_value * block_shape[i] + thread_idx_value
            task_idx = var('task_idx')
            stmts.append(LetStmt(task_idx, simplify(task_idx_value)))
            cond = And(cond, task_idx < task_shape[i])
            task_indices.append(task_idx)
        return stmts, cond, task_indices, Grid(simplify(reduce_product(grid_shape)), simplify(reduce_product(block_shape)))

    @staticmethod
    def get_block_shape(rank: int):
        assert rank >= 1
        if rank == 1:
            return [convert(256)]
        elif rank == 2:
            return [convert(16), convert(16)]
        elif rank == 3:
            return [convert(8), convert(8), convert(8)]
        else:
            return [convert(1) for _ in range(rank - 3)] + [convert(8), convert(8), convert(8)]

    def implement(self, task: Task, match: Mapping[Node, Node]) -> IRModule:
        assert isinstance(task.worker, Grid)
        assert isinstance(task.compute, TensorCompute)
        func_param_vars = [Var(param.name, tp) for param, tp in zip(task.params, task.params_type)]
        task_shape = task.compute.shape

        param2type = {p: t for p, t in zip(task.params, task.params_type)}
        param2arg = {p: arg for p, arg in zip(task.params, func_param_vars)}

        rank = len(task_shape)
        block_shape = self.get_block_shape(rank)
        grid_shape = [(a + b - 1) / b for a, b in zip(task_shape, block_shape)]
        statements, cond, task_indices, worker = self.grid_thread_map(grid_shape, block_shape, task_shape)
        statements.append(IfStmt(cond))

        index_params = [scalar_input(f'i{i}', 'int32') for i in range(rank)]
        rmap = {axis: task_index for axis, task_index in zip(task.compute.axes, index_params)}
        param2arg.update({param: arg for param, arg in zip(index_params, task_indices)})
        param2type.update({param: scalar_type('int32') for param in index_params})

        subtask_name = task.name + '_thread'

        subtask_compute = rewrite(task.compute.value, rmap)

        subtask_params = collect(subtask_compute, [ScalarInput, TensorInput])

        subtask_params += [subtask_compute]
        param2type[subtask_compute] = ReferenceType(infer_type(subtask_compute))
        param2arg[subtask_compute] = TensorElement(param2arg[task.compute], task_indices)

        subtask = Task(subtask_name, subtask_compute, subtask_params, [param2type[p] for p in subtask_params], Thread())

        subtask_module = implement(subtask)
        subtask_func = subtask_module.lookup(subtask_name)

        subtask_args = [param2arg[p] for p in subtask_params]

        statements.append(EvaluateStmt(Call(subtask_module.lookup_var(subtask_func.name), subtask_args)))

        body = concat_stmts(statements)
        func = Function(task.name + '.grid', func_param_vars, body, VoidType(), [], [thread_idx(), block_idx()], {'worker': worker})
        module = IRModule(task=task, funcs={func.name: func})
        module.include(subtask_module)
        return module
