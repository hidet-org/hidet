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
from typing import List

from hidet.ir import IRModule
from hidet.ir.builders import FunctionBuilder
from hidet.ir.expr import scalar_var, convert, Expr, LogicalAnd, cast
from hidet.ir.mapping import TaskMapping
from hidet.ir.primitives import block_idx, thread_idx
from hidet.ir.compute import ReduceOperation
from hidet.ir.stmt import AssignStmt, BufferStoreStmt, DeclareStmt
from hidet.ir.type import data_type
from hidet.ir.utils import index_deserialize
from hidet.graph.ops.definitions.reduce import ReduceTask
from hidet.graph.ops.schedules.common import params_from_task
from hidet.utils import prod
from hidet.transforms.tools import fuse_and_pack
from .common import warp_reduce


def merge_indices(grid_indices: List[Expr], reduce_indices: List[Expr], reduce_dims: List[int]) -> List[Expr]:
    indices = []
    grid_indices = list(reversed(grid_indices))
    reduce_indices = list(reversed(reduce_indices))
    for i in range(len(grid_indices) + len(reduce_indices)):
        if i in reduce_dims:
            indices.append(reduce_indices.pop())
        else:
            indices.append(grid_indices.pop())
    return indices


def cuda_schedule_reduce_by_warp_reduce(task: ReduceTask) -> IRModule:
    x, y = task.inputs[0], task.outputs[0]

    shape: List[int] = x.const_shape()
    dims = task.dims

    grid_shape = [v for i, v in enumerate(shape) if i not in dims]
    reduce_shape = [shape[i] for i in dims]

    grid_layout = TaskMapping.row_major(task_shape=grid_shape)

    warp_size = 32
    reduce_extent = prod(reduce_shape)
    warp_extent = (reduce_extent + warp_size - 1) // warp_size
    block_layout = TaskMapping.full_layout([warp_extent]) * TaskMapping.row_major([warp_size])

    x_dtype = task.inputs[0].ttype.dtype
    accumulate_dtype = task.attributes['accumulate_dtype']

    with FunctionBuilder(
        name=task.name + '_grid',
        kind='cuda_kernel',
        grid_dim=grid_layout.num_workers,
        block_dim=block_layout.num_workers,
        label='reduce schedule',
    ) as fb:
        # params
        params = params_from_task(task)
        x, y = params
        fb.extend_params(params)

        # local variables
        ro = ReduceOperation.from_name(task.reduce_type)
        rv = scalar_var('rv', accumulate_dtype)  # rv stands for reduce value
        fb += DeclareStmt(rv, init=ro.initial_value(data_type(accumulate_dtype)))

        # body
        grid_indices = grid_layout.worker2task(block_idx())[0]

        # get the reduced value along reduce dimensions
        for (r,) in block_layout.worker2task(thread_idx()):
            with fb.if_then(r < reduce_extent):
                reduce_indices = index_deserialize(r, shape=reduce_shape)
                input_indices = merge_indices(grid_indices, reduce_indices, reduce_dims=task.dims)
                fb += AssignStmt(rv, ro.combine(rv, x[input_indices]))

        fb += warp_reduce(rv, op=ro.combine)
        fb += AssignStmt(rv, ro.finalize(acc=rv, size=reduce_extent))

        # write back
        for (r,) in block_layout.worker2task(thread_idx()):
            with fb.if_then(r < reduce_extent):
                reduce_indices = index_deserialize(r, shape=reduce_shape)
                with fb.if_then(LogicalAnd.join_list([reduce_index.equals(0) for reduce_index in reduce_indices])):
                    reduce_indices = [convert(0) for _ in task.dims]
                    if task.keep_dim:
                        output_indices = merge_indices(grid_indices, reduce_indices, reduce_dims=task.dims)
                    else:
                        output_indices = grid_indices
                    fb += BufferStoreStmt(y, output_indices, cast(rv, x_dtype))

        fb.set_body(fb.finish())
    func = fb.get()
    ir_module = IRModule(funcs={func.name: func}, task=task)
    return fuse_and_pack(ir_module, func, task)


def cuda_schedule_reduce_by_default(task: ReduceTask) -> IRModule:
    x, y = task.inputs[0], task.outputs[0]

    shape: List[int] = x.const_shape()
    dims = task.dims

    remain_shape = [v for i, v in enumerate(shape) if i not in dims]
    reduce_shape = [shape[i] for i in dims]
    reduce_extent = prod(reduce_shape)

    block_size = 256
    remain_layout = TaskMapping.row_major(remain_shape)
    reduce_layout = TaskMapping.full_layout(reduce_shape)

    grid_size = (remain_layout.num_workers + block_size - 1) // block_size

    x_dtype = task.inputs[0].ttype.dtype
    accumulate_dtype = task.attributes['accumulate_dtype']

    with FunctionBuilder(
        name=task.name + '_grid', kind='cuda_kernel', grid_dim=grid_size, block_dim=block_size, label='reduce schedule'
    ) as fb:
        # params
        params = params_from_task(task)
        x, y = params
        fb.extend_params(params)

        # local variables
        ro = ReduceOperation.from_name(task.reduce_type)
        rv = scalar_var('rv', accumulate_dtype)  # rv stands for reduce value
        fb += DeclareStmt(rv, init=ro.initial_value(data_type(accumulate_dtype)))

        # body
        remain_indices = remain_layout.worker2task(thread_idx() + block_idx() * block_size)[0]
        with fb.if_then(
            LogicalAnd.join_list([remain_index < remain_shape[i] for i, remain_index in enumerate(remain_indices)])
        ):
            # get the reduced value along reduce dimensions
            for reduce_indices in reduce_layout.worker2task(0):
                input_indices = merge_indices(remain_indices, reduce_indices, reduce_dims=task.dims)
                fb += AssignStmt(rv, ro.combine(rv, x[input_indices]))
            fb += AssignStmt(rv, ro.finalize(acc=rv, size=reduce_extent))

            # write back
            reduce_indices = [convert(0) for _ in reduce_shape]
            if task.keep_dim:
                output_indices = merge_indices(remain_indices, reduce_indices, reduce_dims=task.dims)
            else:
                output_indices = remain_indices
            fb += BufferStoreStmt(y, output_indices, cast(rv, x_dtype))

    func = fb.get()
    ir_module = IRModule(funcs={func.name: func}, task=task)
    return fuse_and_pack(ir_module, func, task)
