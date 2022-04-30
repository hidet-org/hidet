from hidet.tos.ops.schedules.common import expand_loop
from hidet.ir import IRModule
from hidet.ir.builders import FunctionBuilder, StmtBuilder
from hidet.ir.dialects.compute import TensorNode
from hidet.ir.expr import Var
from hidet.ir.functors import rewrite
from hidet.ir.layout import TaskLayout
from hidet.ir.primitives import block_idx, thread_idx
from hidet.ir.stmt import BufferStoreStmt
from hidet.ir.task import Task
from hidet.ir.functors import inline_compute

from ..common import params_from_task


def generic_cuda_schedule(task: Task) -> IRModule:
    computation: TensorNode = inline_compute(task.outputs[0], reduce_limit=16)
    block_size = 512
    task_shape = computation.const_shape()
    task_layout = TaskLayout.row_major(task_shape)
    num_blocks = (task_layout.num_workers + block_size - 1) // block_size

    with FunctionBuilder(name=task.name + '_grid', grid_dim=num_blocks, block_dim=block_size, kind='cuda_kernel', label='generic implementer') as fb:
        # params
        params = params_from_task(task)
        param_map = {param: var for param, var in zip(task.inputs + task.outputs, params)}
        fb.extend_params(params)
        scalar_value = rewrite(computation.grid_compute.value, param_map)  # replace TensorInput to function parameter
        assert len(task.outputs) == 1
        out = param_map[task.outputs[0]]
        # body
        sb = StmtBuilder()
        worker_idx = block_idx() * block_size + thread_idx()
        with sb.if_then(worker_idx < task_layout.num_workers):
            with sb.for_task(worker_index=worker_idx, task_layout=task_layout) as tasks:
                buffer_map = {}
                for axes_values in tasks:
                    remap = {axis: value for axis, value in zip(computation.grid_compute.axes, axes_values)}
                    stmt, value, new_buffer_map = expand_loop(rewrite(scalar_value, remap), input_map=buffer_map)
                    buffer_map.update(new_buffer_map)
                    sb += stmt
                    sb += BufferStoreStmt(out, axes_values, value)
            fb.extend_local_vars(list(buffer_map.values()))
        fb.set_body(sb.finish())
    func = fb.get()
    return IRModule(funcs={func.name: func}, task=task)

