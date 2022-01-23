from typing import Mapping

from hidet.implement.implementer import Implementer, register_impl
from hidet.ir.builders import FunctionBuilder, StmtBuilder
from hidet.ir.dialects.compute import TensorInput, TensorCompute
from hidet.ir.dialects.pattern import TaskPattern, OptionalPattern
from hidet.ir.expr import Constant, TensorElement, var, Var
from hidet.ir.func import IRModule
from hidet.ir.layout import TaskLayout, get_task_layouts
from hidet.ir.node import Node
from hidet.ir.primitives import thread_idx
from hidet.ir.stmt import BufferStoreStmt
from hidet.ir.task import Task, ThreadBlock
from hidet.ir.type import scalar_type, TensorType


@register_impl('cuda_block_transfer_2d_implementer')
class CudaBlockTransfer2dImplementer(Implementer):
    def __init__(self):
        self.task_shape = Constant(None, dtype=scalar_type('int32')), Constant(None, dtype=scalar_type('int32'))
        self.block_size = Constant(None, dtype=scalar_type('int32'))

        self.input = TensorInput('in', None, None)
        self.axes = [var('i'), var('j')]
        self.value = TensorElement(self.input, self.axes)
        self.computation = TensorCompute('out',
                                         shape=self.task_shape,
                                         axes=self.axes,
                                         value=self.value
                                         )
        self.input_type = TensorType()
        self.output_type = TensorType()
        self.task_layout = OptionalPattern(TaskLayout())

        self.pattern = TaskPattern(
            compute_pattern=self.computation,
            required_params=[self.input, self.computation],
            required_param_types=[self.input_type, self.output_type],
            allow_tensor_extra_params=False,
            worker=ThreadBlock(block_dim=self.block_size, task_layout=self.task_layout)
        )

    def priority(self) -> int:
        return 0

    def task_pattern(self) -> TaskPattern:
        return self.pattern

    def implement(self, task: Task, match: Mapping[Node, Node]) -> IRModule:
        block_size = int(match[self.block_size])
        task_shape = [int(match[v]) for v in self.task_shape]
        task_layout = match[self.task_layout]
        ir_module = IRModule(task=task)
        if task_layout is not None:
            ir_module.include(self.implement_for_task_layout(task, match, task_layout))
        else:
            task_layouts = get_task_layouts(valid_num_workers=block_size, task_shape=task_shape)
            for task_layout in task_layouts:
                ir_module.include(self.implement_for_task_layout(task, match, task_layout))
        return ir_module

    def implement_for_task_layout(self, task: Task, match: Mapping[Node, Node], task_layout: TaskLayout) -> IRModule:
        ir_module = IRModule()

        input_type: TensorType = match[self.input_type]
        output_type: TensorType = match[self.output_type]
        with FunctionBuilder(task.name, attrs={'worker': task.worker}) as fb:
            # params
            input_var = Var('in', input_type)
            output_var = Var('out', output_type)
            fb.extend_params([input_var, output_var])

            # body
            sb = StmtBuilder()
            for task_index in task_layout.worker2task(thread_idx()):
                sb.append(BufferStoreStmt(output_var, task_index, TensorElement(input_var, task_index)))

            fb.set_body(sb.finish())
        func = fb.get()
        ir_module.add(func.name, func)
        return ir_module
