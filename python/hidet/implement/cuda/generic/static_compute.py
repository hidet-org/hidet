import contextlib
import sys
from typing import Mapping, Any, List, Tuple, Union

from hidet.implement.implementer import register_impl, Implementer, NotSupportedError, Schedule
from hidet.ir import IRModule
from hidet.ir.builders import FunctionBuilder, StmtBuilder
from hidet.ir.dialects.compute import TensorCompute, ReduceCompute, compute
from hidet.ir.dialects.pattern import TaskPattern, any_const_ints, any_scalar_expr, int_vars, StringPattern, AnyExpr, TensorComputePattern
from hidet.ir.expr import Var, Constant, And, if_then_else, Equal, scalar_var
from hidet.ir.stmt import BufferStoreStmt, IfStmt, AssignStmt
from hidet.ir.functors import rewrite
from hidet.ir.layout import TaskLayout, DataLayout
from hidet.ir.node import Node
from hidet.ir.primitives import block_idx, thread_idx, syncthreads, printf, cuda_max
from hidet.ir.task import Task, Grid
from hidet.ir.type import TensorType
from hidet.utils import prod, cuda
from hidet.implement.common import VirtualTensor, pattern2matched, expand_loop
from hidet.utils.info import float_type_min_value


class Pattern:
    def __init__(self):
        self.value_expr = AnyExpr()
        self.compute = TensorComputePattern(
            rank=None,
            allow_dynamic_axis=False,
            value=self.value_expr
        )
        self.task_pattern = TaskPattern(
            compute_pattern=self.compute,
            worker=Grid()
        )


@register_impl('cuda_static_compute_implementer')
class CudaStaticComputeImplementer(Implementer):
    def __init__(self):
        self.pattern = Pattern()

    def priority(self) -> int:
        return 0

    def task_pattern(self) -> TaskPattern:
        return self.pattern.task_pattern

    def implement(self, task: Task, match: Mapping[Node, Any]) -> IRModule:
        p = pattern2matched(self.pattern, match)
        computation: TensorCompute = p.compute
        block_size = 512
        task_shape = [int(v) for v in computation.shape]
        task_layout = TaskLayout.row_major(task_shape)
        num_blocks = (task_layout.num_workers + block_size - 1) // block_size

        with FunctionBuilder(name=task.name + '_grid', worker=Grid(num_blocks, block_size), label='generic implementer') as fb:
            # params
            params = [Var(param.name, param_type) for param, param_type in zip(task.params, task.params_type)]
            param_map = {param: var for param, var in zip(task.params, params)}
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

