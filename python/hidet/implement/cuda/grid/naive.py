from typing import Mapping
from functools import reduce
import operator

from hidet.ir.node import Node
from hidet.ir.type import scalar_type
from hidet.ir.expr import Var, IntVar, Call, convert
from hidet.ir.stmt import LetStmt, IfStmt, EvaluateStmt, concat_stmts, SeqStmt, BufferStoreStmt
from hidet.ir.task import Task, Grid, Thread
from hidet.ir.func import IRModule, Function
from hidet.ir.dialects.compute import TensorInput, ScalarInput, TensorCompute
from hidet.ir.dialects.pattern import TaskPattern, TensorComputePattern
from hidet.ir.dialects.lowlevel import VoidType
from hidet.ir.functors import rewrite, infer_type, collect, simplify
from hidet.implement.implementer import Implementer, implement, register_impl


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

    def implement(self, task: Task, match: Mapping[Node, Node]) -> IRModule:
        assert isinstance(task.worker, Grid)
        assert isinstance(task.compute, TensorCompute)
        func_param_vars = [Var(param.name, tp) for param, tp in zip(task.params, task.params_type)]
        shape = task.compute.shape

        block_dim = 256
        grid_dim = simplify((reduce(operator.mul, shape, convert(1)) + block_dim - 1) // block_dim)

        param2type = {p: t for p, t in zip(task.params, task.params_type)}
        param2arg = {p: arg for p, arg in zip(task.params, func_param_vars)}

        statements = []
        rmap = {}
        axes_vars = []
        thread_index = IntVar()
        statements.append(LetStmt(thread_index, IntVar('blockIdx.x', grid_dim) * block_dim + IntVar('threadIdx.x', block_dim)))
        for i in range(len(shape)):
            p = reduce(operator.mul, shape[i + 1:], convert(1))
            axis_var = IntVar()
            axes_vars.append(axis_var)
            statements.append(LetStmt(axis_var, (thread_index // p) % shape[i]))
            si = ScalarInput(None, 'int32')
            rmap[task.compute.axes[i]] = si
            param2arg[si] = axis_var
            param2type[si] = scalar_type('int32')

        for i in range(len(shape)):
            statements.append(IfStmt(axes_vars[i] < task.compute.axes[i].extent))

        subtask_name = task.name + '.thread'

        subtask_compute = rewrite(task.compute.value, rmap)

        subtask_params = collect(subtask_compute, [ScalarInput, TensorInput])

        subtask_ret_var = Var('out', infer_type(subtask_compute))
        subtask_params += [subtask_compute]
        param2type[subtask_compute] = infer_type(subtask_compute)
        param2arg[subtask_compute] = subtask_ret_var

        subtask = Task(subtask_name, subtask_compute, subtask_params, [param2type[p] for p in subtask_params], Thread())

        subtask_module = implement(subtask)
        subtask_func = subtask_module.lookup(subtask_name)
        subtask_func_var = subtask_module.lookup_var(subtask_func.name)

        subtask_args = [param2arg[p] for p in subtask_params]

        inner_stmts = SeqStmt([])
        inner_stmts.append(EvaluateStmt(Call(subtask_func_var, subtask_args)))
        inner_stmts.append(BufferStoreStmt(param2arg[task.compute], axes_vars, subtask_ret_var))
        statements.append(inner_stmts)

        body = concat_stmts(statements)
        func = Function(task.name + '.grid', func_param_vars, body, VoidType(), [subtask_ret_var], {'worker': Grid(grid_dim, block_dim)})
        module = IRModule({func.name: func})
        module.include(subtask_module)
        return module


