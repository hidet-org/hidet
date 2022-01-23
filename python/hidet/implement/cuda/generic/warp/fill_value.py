from itertools import product
from typing import Mapping

from hidet.implement.implementer import Implementer, register_impl
from hidet.ir.builders import FunctionBuilder, StmtBuilder
from hidet.ir.dialects.compute import TensorInput, TensorCompute
from hidet.ir.dialects.pattern import TaskPattern
from hidet.ir.expr import Constant, TensorElement, var, tensor_var, convert, Var
from hidet.ir.func import IRModule
from hidet.ir.node import Node
from hidet.ir.stmt import BufferStoreStmt, LetStmt
from hidet.ir.task import Task, Warp
from hidet.ir.type import scalar_type, TensorType, ScalarType
from hidet.ir.primitives import thread_idx
from hidet.ir.layout import TaskLayout


@register_impl('cuda_warp_fill_value_implementer')
class CudaWarpFillValueImplementer(Implementer):
    def __init__(self):
        self.task_shape = [Constant(None, dtype=scalar_type('int32')), Constant(None, dtype=scalar_type('int32'))]

        self.axes = [var('i'), var('j')]
        self.value = Constant(None)
        self.computation = TensorCompute('out', shape=self.task_shape, axes=self.axes, value=self.value)
        self.output_type = TensorType()
        self.task_layout = TaskLayout()

        self.pattern = TaskPattern(
            compute_pattern=self.computation,
            required_params=[self.computation],
            required_param_types=[self.output_type],
            allow_tensor_extra_params=False,
            worker=Warp(self.task_layout)
        )

    def priority(self) -> int:
        return 0

    def task_pattern(self) -> TaskPattern:
        return self.pattern

    def implement(self, task: Task, match: Mapping[Node, Node]) -> IRModule:
        ir_module = IRModule(task=task)
        value = match[self.value]
        task_layout: TaskLayout = match[self.task_layout]
        output_type = match[self.output_type]

        with FunctionBuilder(task.name, attrs={'worker': Warp()}) as fb:
            # params
            output_var = Var('out', output_type)
            fb.extend_params([output_var])

            # body
            sb = StmtBuilder()
            lane_id = var('lane_id')
            sb.append(LetStmt(lane_id, thread_idx() % 32))
            sb.enter_body()
            for task_index in task_layout.worker2task(lane_id):
                sb.append(BufferStoreStmt(output_var, task_index, value))
            sb.exit_body()
            fb.set_body(sb.finish())
        func = fb.get()
        ir_module.add(func.name, func)
        return ir_module

