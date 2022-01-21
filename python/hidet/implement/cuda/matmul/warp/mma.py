from typing import Mapping, List, Tuple
from itertools import product

from hidet.implement.implementer import Implementer, register_impl
from hidet.ir.dialects.compute import TensorInput, TensorCompute, reduce_sum
from hidet.ir.dialects.pattern import TaskPattern
from hidet.ir.builders import FunctionBuilder, StmtBuilder
from hidet.ir.expr import Expr, Constant, TensorElement, var, tensor_var, convert
from hidet.ir.stmt import AssignStmt, BufferStoreStmt, ForStmt, LetStmt
from hidet.ir.func import IRModule
from hidet.ir.node import Node
from hidet.ir.task import Task, ThreadBlock, Warp
from hidet.ir.type import scalar_type, TensorType, ScalarType, Scope, RegisterScope, tensor_type
from hidet.ir.primitives import thread_idx
from hidet.implement.cuda.layout import get_task_layouts, TaskLayout
from hidet.implement.search_space import AtomSpace, SpaceChoice
from hidet.implement.implementer import NotSupportedError


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
                                         value=self.reduce_value
                                         )
        self.A_dtype = scalar_type(None)
        self.B_dtype = scalar_type(None)
        self.C_dtype = scalar_type(None)
        self.A_scope = RegisterScope()
        self.B_scope = RegisterScope()
        self.C_scope = RegisterScope()
        self.A_type = TensorType(self.A_scope, self.A_dtype)
        self.B_type = TensorType(self.B_scope, self.B_dtype)
        self.C_type = TensorType(self.C_scope, self.C_dtype)

        self.pattern = TaskPattern(
            compute_pattern=self.computation,
            required_params=[self.A, self.B, self.computation],
            required_param_types=[self.A_type, self.B_type, self.C_type],
            allow_tensor_extra_params=False,
            worker=Warp()
        )

    def priority(self) -> int:
        return 0

    def task_pattern(self) -> TaskPattern:
        return self.pattern

    def implement(self, task: Task, match: Mapping[Node, Node]) -> IRModule:
        task_m, task_n, task_k = int(match[self.task_m]), int(match[self.task_n]), int(match[self.task_k])
        A_scope: RegisterScope = match[self.A_scope]
        B_scope: RegisterScope = match[self.B_scope]
        C_scope: RegisterScope = match[self.C_scope]
        ir_module = IRModule()
        with FunctionBuilder(task.name, attrs={'worker': Warp()}) as fb:
            # params
            A = tensor_var('A', shape=A_scope.local_shape, scope=A_scope, dtype=match[self.A_dtype])
            B = tensor_var('B', shape=B_scope.local_shape, scope=B_scope, dtype=match[self.B_dtype])
            C = tensor_var('C', shape=C_scope.local_shape, scope=C_scope, dtype=match[self.C_dtype])
            fb.extend_params([A, B, C])
            # body
            sb = StmtBuilder()
            block_tid = thread_idx()
            lane_id = var('lane_id')
            sb.append(LetStmt(lane_id, block_tid % 32))
            sb.enter_body()
            local_m, local_n = C_scope.local_shape
            for C_locals in product(range(local_m), range(local_n)):
                global_i, global_j = C_scope.local2global(lane_id, *C_locals)
                k = var('k')
                sb.append(ForStmt(k, task_k))
                with sb.for_body():
                    A_locals = [var('A_local_p'), var('A_local_q')]
                    B_locals = [var('B_local_p'), var('B_local_q')]
                    A_local_vals = A_scope.global2local(global_i, k)
                    B_local_vals = B_scope.global2local(k, global_j)
                    sb.append(LetStmt(A_locals[0], A_local_vals[0]))
                    sb.enter_body()
                    sb.append(LetStmt(A_locals[1], A_local_vals[1]))
                    sb.enter_body()
                    sb.append(LetStmt(B_locals[0], B_local_vals[0]))
                    sb.enter_body()
                    sb.append(LetStmt(B_locals[1], B_local_vals[1]))
                    sb.enter_body()
                    c_val = TensorElement(C, C_locals)
                    a_val = TensorElement(A, A_locals)
                    b_val = TensorElement(B, B_locals)
                    sb.append(BufferStoreStmt(C, C_locals, c_val + a_val * b_val))
                    sb.exit_body(num_scopes=4)
            sb.exit_body()  # let lane_id
            fb.set_body(sb.finish())
        func = fb.get()
        ir_module.add(func.name, func)
        return ir_module

