from typing import Mapping

from hidet.implement.implementer import Implementer, register_impl
from hidet.ir.builders import FunctionBuilder, StmtBuilder
from hidet.ir.dialects.compute import TensorInput, TensorCompute, reduce_sum
from hidet.ir.dialects.pattern import TaskPattern
from hidet.ir.expr import Constant, var, Var
from hidet.ir.func import IRModule
from hidet.ir.layout import TaskLayout
from hidet.ir.node import Node
from hidet.ir.primitives import thread_idx
from hidet.ir.stmt import BufferStoreStmt, ForStmt, LetStmt
from hidet.ir.task import Task, Warp
from hidet.ir.type import TensorType


@register_impl('cuda_warp_mma_implementer')
class CudaWarpMmaImplementer(Implementer):
    def __init__(self):
        self.A = TensorInput('A')
        self.B = TensorInput('B')
        self.axis_i = var('i')
        self.axis_j = var('j')
        self.axis_k = var('k')
        self.task_m = Constant(None, dtype='int32')
        self.task_n = Constant(None, dtype='int32')
        self.task_k = Constant(None, dtype='int32')
        self.reduce_value = reduce_sum(self.A[self.axis_i, self.axis_k] * self.B[self.axis_k, self.axis_j], self.axis_k, self.task_k)
        self.computation = TensorCompute(name='C',
                                         shape=[self.task_m, self.task_n],
                                         axes=[self.axis_i, self.axis_j],
                                         value=self.reduce_value)
        self.A_type = TensorType()
        self.B_type = TensorType()
        self.C_type = TensorType()

        self.task_layout: TaskLayout = TaskLayout()

        self.pattern = TaskPattern(
            compute_pattern=self.computation,
            required_params=[self.A, self.B, self.computation],
            required_param_types=[self.A_type, self.B_type, self.C_type],
            allow_tensor_extra_params=False,
            worker=Warp(task_layout=self.task_layout)
        )

    def priority(self) -> int:
        return 0

    def task_pattern(self) -> TaskPattern:
        return self.pattern

    def implement(self, task: Task, match: Mapping[Node, Node]) -> IRModule:
        task_m, task_n, task_k = int(match[self.task_m]), int(match[self.task_n]), int(match[self.task_k])
        ir_module = IRModule(task=task)
        A_type = match[self.A_type]
        B_type = match[self.B_type]
        C_type = match[self.C_type]
        task_layout: TaskLayout = match[self.task_layout]
        with FunctionBuilder(task.name, attrs={'worker': Warp()}) as fb:
            # params
            A = Var('A', A_type)
            B = Var('B', B_type)
            C = Var('C', C_type)
            fb.extend_params([A, B, C])

            # body
            sb = StmtBuilder()
            with sb.let('lane_id', thread_idx() % 32) as lane_id:
                for i, j in task_layout.worker2task(lane_id):
                    with sb.for_loop('k', task_k) as k:
                        sb.append(BufferStoreStmt(C, [i, j], C[i, j] + A[i, k] * B[k, j]))
            fb.set_body(sb.finish())
        func = fb.get()
        ir_module.add(func.name, func)
        return ir_module

