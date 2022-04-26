from hidet.implement.common import expand_loop
from hidet.ir import IRModule
from hidet.ir.builders import FunctionBuilder, StmtBuilder
from hidet.ir.dialects.compute import TensorCompute
from hidet.ir.expr import Var
from hidet.ir.functors import rewrite
from hidet.ir.layout import TaskLayout
from hidet.ir.primitives import block_idx, thread_idx
from hidet.ir.stmt import BufferStoreStmt
from hidet.ir.task import Grid
from hidet.tos.task import Task
from hidet.ir.functors import inline_compute


def generic_cuda_schedule(task: Task) -> IRModule:
    computation: TensorCompute = inline_compute(task.outputs[0], reduce_limit=16)
    block_size = 512
    task_shape = computation.const_shape()
    task_layout = TaskLayout.row_major(task_shape)
    num_blocks = (task_layout.num_workers + block_size - 1) // block_size

    with FunctionBuilder(name=task.name + '_grid', worker=Grid(num_blocks, block_size), label='generic implementer') as fb:
        # params
        params = [Var(param.name, param.data_type) for param in task.parameters]
        param_map = {param: var for param, var in zip(task.parameters, params)}
        fb.extend_params(params)
        scalar_value = rewrite(computation.value, param_map)  # replace TensorInput to function parameter
        out = param_map[task.compute]
        # body
        sb = StmtBuilder()
        worker_idx = block_idx() * block_size + thread_idx()
        with sb.if_then(worker_idx < task_layout.num_workers):
            with sb.for_task(worker_index=worker_idx, task_layout=task_layout) as tasks:
                buffer_map = {}
                for axes_values in tasks:
                    remap = {axis: value for axis, value in zip(computation.axes, axes_values)}
                    stmt, value, new_buffer_map = expand_loop(rewrite(scalar_value, remap), input_map=buffer_map)
                    buffer_map.update(new_buffer_map)
                    sb += stmt
                    sb += BufferStoreStmt(out, axes_values, value)
            fb.extend_local_vars(list(buffer_map.values()))
        fb.set_body(sb.finish())
    func = fb.get()
    return IRModule(funcs={func.name: func}, task=task)

