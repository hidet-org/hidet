from typing import List

from hidet.ir import IRModule
from hidet.ir.builders import FunctionBuilder, StmtBuilder
from hidet.ir.expr import scalar_var, if_then_else, tensor_var
from hidet.ir.layout import TaskLayout
from hidet.ir.primitives import expf, block_idx, thread_idx, cuda_max
from hidet.ir.stmt import AssignStmt, BufferStoreStmt
from hidet.ir.task import Grid
from hidet.tos.ops.definitions.softmax import SoftmaxTask
from hidet.tos.ops.schedules.common import params_from_task, inputs_from_task, outputs_from_task, write_output
from .common import warp_reduce


def softmax_cuda_schedule(task: SoftmaxTask) -> IRModule:
    shape: List[int] = task.x_shape
    axis = task.axis

    other_shape = shape[:axis] + shape[axis+1:]
    grid_layout = TaskLayout.row_major(task_shape=other_shape)

    warp_size = 32
    reduce_extent = shape[axis]
    outer_extent = (reduce_extent + warp_size - 1) // warp_size
    block_layout = TaskLayout.full_layout([outer_extent]) * TaskLayout.row_major([warp_size])

    x_dtype = task.inputs[0].data_type.scalar_type

    with FunctionBuilder(
            name=task.name + '_grid',
            worker=Grid(grid_layout.num_workers, block_layout.num_workers),
            label='softmax schedule'
    ) as fb:
        # params
        params = params_from_task(task)
        x = inputs_from_task(task, params)[0]
        y = outputs_from_task(task, params)[0]
        fb.extend_params(params)

        # local variables
        buf = tensor_var('buf', shape=[outer_extent], scope='register', dtype=x_dtype)
        rv = scalar_var('rv', x_dtype)  # rv stands for reduce value
        fb.extend_local_vars([rv, buf])

        # body
        sb = StmtBuilder()

        # get the max value along c dimension
        sb += AssignStmt(rv, -1e30)
        other_indices = grid_layout.worker2task(block_idx())[0]
        for r, in block_layout.worker2task(thread_idx()):
            with sb.if_then(r < reduce_extent):
                sb += BufferStoreStmt(buf, [r], x[other_indices[:axis] + (r,) + other_indices[axis:]])
                sb += AssignStmt(rv, cuda_max(rv, buf[r]))
        sb += warp_reduce(rv, cuda_max)

        # calculate exp(v-max)
        for r, in block_layout.worker2task(thread_idx()):
            sb += AssignStmt(buf[r], expf(buf[r] - rv))

        # calculate sum(exp(v-max))
        sb += AssignStmt(rv, 0.0)
        for r, in block_layout.worker2task(thread_idx()):
            sb += AssignStmt(rv, rv + if_then_else(r < reduce_extent, buf[r], 0.0))
        sb += warp_reduce(rv, lambda a, b: a + b)

        # calculate exp(v-max) / sum(exp(vv-max))
        for r, in block_layout.worker2task(thread_idx()):
            with sb.if_then(r < reduce_extent):
                sb += write_output(y, other_indices[:axis] + (r,) + other_indices[axis:], buf[r] / rv, task, params)

        fb.set_body(sb.finish())
    func = fb.get()
    return IRModule(funcs={func.name: func}, task=task)
