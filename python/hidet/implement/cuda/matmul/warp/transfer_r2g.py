from typing import Mapping, List, Tuple
from itertools import product

from hidet.implement.implementer import Implementer, register_impl
from hidet.ir.dialects.compute import TensorInput, TensorCompute
from hidet.ir.dialects.pattern import TaskPattern
from hidet.ir.builders import FunctionBuilder, StmtBuilder
from hidet.ir.expr import Expr, Constant, TensorElement, var, tensor_var
from hidet.ir.stmt import AssignStmt, BufferStoreStmt, LetStmt
from hidet.ir.func import IRModule
from hidet.ir.node import Node
from hidet.ir.task import Task, ThreadBlock, Warp
from hidet.ir.type import scalar_type, TensorType, ScalarType, Scope, RegisterScope
from hidet.ir.primitives import thread_idx
from hidet.implement.cuda.layout import get_task_layouts, TaskLayout
from hidet.implement.search_space import AtomSpace, SpaceChoice
from hidet.implement.implementer import NotSupportedError


@register_impl('cuda_warp_transfer_r2g_implementer')
class CudaWarpTransferR2GImplementer(Implementer):
    def __init__(self):
        self.shape = [Constant(None, dtype=scalar_type('int32')), Constant(None, dtype=scalar_type('int32'))]
        self.out_strides = [Constant(None, dtype=scalar_type('int32')), Constant(None, dtype=scalar_type('int32'))]

        self.input = TensorInput('in', None, None)
        self.axes = [var('i'), var('j')]
        self.computation = TensorCompute('out',
                                         shape=self.shape,
                                         axes=self.axes,
                                         value=TensorElement(self.input, self.axes))
        self.regs_dtype = ScalarType(None)
        self.gmem_dtype = ScalarType(None)
        self.reg_scope = RegisterScope()
        self.input_type = TensorType(self.reg_scope, self.regs_dtype)
        self.output_type = TensorType('global', self.gmem_dtype, strides=self.out_strides)

        self.pattern = TaskPattern(
            compute_pattern=self.computation,
            required_params=[self.input, self.computation],
            required_param_types=[self.input_type, self.output_type],
            allow_tensor_extra_params=False,
            worker=Warp()
        )

    def priority(self) -> int:
        return 0

    def task_pattern(self) -> TaskPattern:
        return self.pattern

    def implement(self, task: Task, match: Mapping[Node, Node]) -> IRModule:
        ir_module = IRModule()

        shape = [match[v] for v in self.shape]
        out_strides = [match[v] for v in self.out_strides]
        reg_scope: RegisterScope = match[self.reg_scope]
        with FunctionBuilder(task.name, attrs={'worker': Warp()}) as fb:
            # params
            regs = tensor_var('regs', shape=reg_scope.local_shape, scope=reg_scope, dtype=match[self.regs_dtype])
            gmem = tensor_var('gmem', shape=shape, scope='global', dtype=match[self.gmem_dtype], strides=out_strides)
            fb.extend_params([regs, gmem])
            # body
            sb = StmtBuilder()
            block_tid = thread_idx()
            lane_id = var('lane_id')
            sb.append(LetStmt(lane_id, block_tid % 32))
            sb.enter_body()
            local_m, local_n = reg_scope.local_shape
            for regs_locals in product(range(local_m), range(local_n)):
                regs_globals = reg_scope.local2global(lane_id, *regs_locals)
                sb.append(BufferStoreStmt(gmem, regs_globals, TensorElement(regs, regs_locals)))
            sb.exit_body()  # let lane_id
            fb.set_body(sb.finish())
        func = fb.get()
        ir_module.add(func.name, func)
        return ir_module

