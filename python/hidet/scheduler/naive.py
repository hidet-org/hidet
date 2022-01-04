import operator
from functools import reduce
from hidet.core.compute import TensorCompute, ScalarInput, TensorInput
from hidet.core.task import Task
from hidet.core.worker import Grid, Thread
from hidet.ir.expr import Var, IntVar, scalar_var, Call
from hidet.ir.stmt import Stmt, LetStmt, IfStmt, EvaluateStmt, flatten, SeqStmt, BufferStoreStmt
from hidet.ir.func import IRModule, Function
from hidet.ir.type import ScalarType, VoidType, scalar_type
from hidet.ir.functors import rewrite, infer_type, collect
from hidet.scheduler.common import expand_loop


def naive_scheduler_grid(task: Task) -> IRModule:
    assert isinstance(task.worker, Grid)
    assert isinstance(task.compute, TensorCompute)
    func_param_vars = [Var(param.name, tp) for param, tp in zip(task.params, task.params_type)]
    shape = task.compute.shape

    block_dim = 256
    grid_dim = (reduce(operator.mul, shape, 1) + block_dim - 1) // block_dim

    param2type = {p: t for p, t in zip(task.params, task.params_type)}
    param2arg = {p: arg for p, arg in zip(task.params, func_param_vars)}

    statements = []
    rmap = {}
    axes_vars = []
    thread_index = IntVar()
    statements.append(LetStmt(thread_index, IntVar('blockIdx.x', grid_dim) * block_dim + IntVar('threadIdx.x', block_dim)))
    for i in range(len(shape)):
        p = reduce(operator.mul, shape[i + 1:], 1)
        axis_var = IntVar()
        axes_vars.append(axis_var)
        statements.append(LetStmt(axis_var, (thread_index // p) % shape[i]))
        si = ScalarInput(None, 'int32')
        rmap[task.compute.axes[i]] = si
        param2arg[si] = axis_var
        param2type[si] = scalar_type('int32')

    for i in range(len(shape)):
        statements.append(IfStmt(axes_vars[i] < task.compute.axes[i].extent))

    #
    # get subtask
    #

    subtask_name = task.name + '.thread'

    subtask_compute = rewrite(task.compute.value, rmap)

    subtask_params = collect(subtask_compute, [ScalarInput, TensorInput])

    subtask_ret_var = Var('out', infer_type(subtask_compute))
    subtask_params += [subtask_compute]
    param2type[subtask_compute] = infer_type(subtask_compute)
    param2arg[subtask_compute] = subtask_ret_var

    subtask = Task(subtask_name, subtask_compute, subtask_params, [param2type[p] for p in subtask_params], Thread())

    subtask_module = naive_scheduler_thread(subtask)
    subtask_func = subtask_module.lookup(subtask_name)
    subtask_func_var = subtask_module.lookup_var(subtask_func.name)

    subtask_args = [param2arg[p] for p in subtask_params]

    inner_stmts = SeqStmt([])
    inner_stmts.append(EvaluateStmt(Call(subtask_func_var, subtask_args)))
    inner_stmts.append(BufferStoreStmt(param2arg[task.compute], axes_vars, subtask_ret_var))
    statements.append(inner_stmts)

    body = flatten(statements)
    func = Function(task.name, func_param_vars, body, VoidType(), [subtask_ret_var], {'worker': Grid(grid_dim, block_dim)})
    module = IRModule({func.name: func})
    module.include(subtask_module)
    return module


def naive_scheduler_thread(task: Task) -> IRModule:
    assert isinstance(task.worker, Thread)
    func_param_vars = [Var(param.name, tp) for param, tp in zip(task.params, task.params_type)]
    input_map = {p: v for p, v in zip(task.params, func_param_vars)}
    body, value, new_buffer_map = expand_loop(task.compute, input_map)
    func_locals = new_buffer_map.values()
    func = Function(task.name, func_param_vars, body, VoidType(), func_locals, {'worker': Thread()})
    module = IRModule({func.name: func})
    return module

