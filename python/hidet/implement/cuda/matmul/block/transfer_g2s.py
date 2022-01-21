from typing import Mapping, List, Tuple

from hidet.implement.implementer import Implementer, register_impl
from hidet.ir.dialects.compute import TensorInput, TensorCompute
from hidet.ir.dialects.pattern import TaskPattern
from hidet.ir.builders import FunctionBuilder, StmtBuilder
from hidet.ir.expr import Expr, Constant, TensorElement, var, tensor_var
from hidet.ir.stmt import AssignStmt, BufferStoreStmt
from hidet.ir.func import IRModule
from hidet.ir.node import Node
from hidet.ir.task import Task, ThreadBlock
from hidet.ir.type import scalar_type, TensorType, ScalarType
from hidet.implement.cuda.layout import get_task_layouts, TaskLayout, full_layout, row_major_layout
from hidet.implement.cuda.layout.concrete import WarpLayout4x8
from hidet.implement.search_space import AtomSpace, SpaceChoice
from hidet.ir.primitives import thread_idx


@register_impl('cuda_block_transfer_g2s_implementer')
class CudaBlockTransferG2SImplementer(Implementer):
    def __init__(self):
        self.block_dim = Constant(None, dtype=scalar_type('int32'))
        self.shape = [Constant(None, dtype=scalar_type('int32')), Constant(None, dtype=scalar_type('int32'))]
        self.in_strides = [Constant(None, dtype=scalar_type('int32')), Constant(None, dtype=scalar_type('int32'))]
        self.out_strides = [Constant(None, dtype=scalar_type('int32')), Constant(None, dtype=scalar_type('int32'))]

        self.input = TensorInput('in', None, None)
        self.axes = [var('i'), var('j')]
        self.value = TensorElement(self.input, self.axes)
        self.computation = TensorCompute('out',
                                         shape=self.shape,
                                         axes=self.axes,
                                         value=self.value
                                         )
        self.gmem_dtype = ScalarType(None)
        self.smem_dtype = ScalarType(None)
        self.input_type = TensorType('global', self.gmem_dtype, None, self.in_strides)
        self.output_type = TensorType('shared', self.smem_dtype, None, self.out_strides)

        self.pattern = TaskPattern(
            compute_pattern=self.computation,
            required_params=[self.input, self.computation],
            required_param_types=[self.input_type, self.output_type],
            allow_tensor_extra_params=False,
            worker=ThreadBlock(block_dim=self.block_dim)
        )

    def priority(self) -> int:
        return 0

    def task_pattern(self) -> TaskPattern:
        return self.pattern

    def implement(self, task: Task, match: Mapping[Node, Node]) -> IRModule:
        block_size = int(match[self.block_dim])
        task_m, task_n = [int(match[v]) for v in self.shape]
        space = AtomSpace('layout', choices=get_task_layouts(valid_num_workers=block_size, task_shape=[task_m, task_n]))
        # todo: use search
        # space = AtomSpace('layout', choices=[row_major_layout(32, 8) * full_layout(4, 1),
        #                                      full_layout(1, 4) * row_major_layout(8, 32)])
        space_size = len(space)
        ir_module = IRModule()
        for i in range(space_size):
            try:
                choice_module = self.implement_for_choice(task, match, space[i])
            except (NotImplementedError, AssertionError) as e:
                continue
            else:
                ir_module.include(choice_module)
        return ir_module

    def implement_for_choice(self, task: Task, match: Mapping[Node, Node], choice: SpaceChoice) -> IRModule:
        task_m, task_n = [int(match[v]) for v in self.shape]
        gmem_dtype = match[self.gmem_dtype]
        smem_dtype = match[self.smem_dtype]
        in_strides = [match[v] for v in self.in_strides]
        out_strides = [match[v] for v in self.out_strides]
        layout: TaskLayout = choice.value
        assert tuple(layout.task_shape) == tuple([task_m, task_n])
        with FunctionBuilder(name=task.name) as fb:
            # params
            gmem = tensor_var('gmem', [task_m, task_n], 'global', dtype=gmem_dtype, strides=in_strides)
            smem = tensor_var('smem', [task_m, task_n], 'shared', dtype=smem_dtype, strides=out_strides)
            fb.extend_params([gmem, smem])

            # body
            sb = StmtBuilder()
            thread_tasks: List[Tuple[Expr, Expr]] = layout.worker2task(thread_idx())
            for thread_task in thread_tasks:
                i, j = thread_task
                sb.append(BufferStoreStmt(smem, [i, j], TensorElement(gmem, [i, j])))
            body = sb.finish()
            fb.set_body(body)
            fb.extend_attrs({'worker': task.worker, 'label': layout.expr_text})
        return IRModule({task.name: fb.get()})
