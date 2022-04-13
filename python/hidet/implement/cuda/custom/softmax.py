import contextlib
import sys
from typing import Mapping, Any, List, Tuple, Union

from hidet.implement.implementer import register_impl, Implementer, NotSupportedError, Schedule
from hidet.ir import IRModule
from hidet.ir.builders import FunctionBuilder, StmtBuilder
from hidet.ir.dialects.compute import TensorCompute, ReduceCompute, compute, CustomCompute, TensorInput
from hidet.ir.dialects.pattern import TaskPattern, any_const_int, any_const_ints, any_scalar_expr, int_vars, StringPattern, AnyExpr, TensorComputePattern
from hidet.ir.expr import Var, Constant, And, if_then_else, Equal, scalar_var, tensor_var
from hidet.ir.stmt import Stmt, BufferStoreStmt, IfStmt, AssignStmt, ReturnStmt
from hidet.ir.functors import rewrite
from hidet.ir.layout import TaskLayout, DataLayout
from hidet.ir.node import Node
from hidet.ir.primitives import expf, block_idx, thread_idx, cuda_max, active_mask, shfl_down_sync, shfl_sync
from hidet.ir.task import Task, Grid
from hidet.ir.type import TensorType
from hidet.utils import prod, cuda
from hidet.implement.common import VirtualTensor, pattern2matched, expand_loop
from hidet.utils.info import float_type_min_value


class Pattern:
    def __init__(self):
        self.axis = any_const_int()
        self.compute = CustomCompute(
            name=None,
            identifier='softmax',
            params=None,
            data_type=None,
            attributes={'axis': self.axis}
        )
        self.task_pattern = TaskPattern(
            compute_pattern=self.compute,
            worker=Grid()
        )


@register_impl('cuda_static_softmax_implementer')
class CudaStaticSoftmaxImplementer(Implementer):
    """
    a, b, x, c, d
    ctaid.y <=> x
    ctaid.x
    tid.x 512
    """
    def __init__(self):
        self.pattern = Pattern()

    def priority(self) -> int:
        return 0

    def task_pattern(self) -> TaskPattern:
        return self.pattern.task_pattern

    def implement(self, task: Task, match: Mapping[Node, Any]) -> IRModule:
        p = pattern2matched(self.pattern, match)
        computation: CustomCompute = p.compute
        inputs: List[Union[TensorInput, TensorCompute]] = [p for p in computation.params if isinstance(p, (TensorInput, TensorCompute))]
        assert len(inputs) == len(computation.params) == 1
        data = inputs[0]
        dtype = data.data_type.scalar_type
        axis: int = p.axis
        shape: List[int] = data.const_shape()
        rank: int = len(shape)
        warp_size = 32

        other_shape = [v for i, v in enumerate(shape) if i != axis]
        grid_layout = TaskLayout.row_major(other_shape)

        reduce_extent = shape[axis]
        outer_extent = (reduce_extent + warp_size - 1) // warp_size
        block_layout = TaskLayout.full_layout([outer_extent]) * TaskLayout.row_major([warp_size])

        with FunctionBuilder(name=task.name + '_grid', worker=Grid(grid_dim=grid_layout.num_workers, block_dim=block_layout.num_workers),
                             label=self.__class__.__name__) as fb:
            # params
            params: List[Var] = [Var(param.name, param.data_type) for param in task.params]
            fb.extend_params(params)
            x, y = params

            # local variables
            buf = tensor_var('buf', shape=[outer_extent], scope='register', dtype=dtype, layout=DataLayout.row_major([outer_extent]))
            rv = scalar_var('rv', dtype)  # rv stands for reduction value
            fb.extend_local_vars([rv, buf])

            # body
            sb = StmtBuilder()
            # get the max value along c dimension
            sb += AssignStmt(rv, sys.float_info.min)
            other_indices = grid_layout.worker2task(block_idx())[0]
            for r, in block_layout.worker2task(thread_idx()):
                with sb.if_then(r < reduce_extent):
                    sb += BufferStoreStmt(buf, [r], x[other_indices[:axis] + (r,) + other_indices[axis:]])
                    sb += AssignStmt(rv, cuda_max(rv, buf[r]))
            sb += self.warp_reduce(rv, cuda_max)
            # calculate exp(v-max)
            for r, in block_layout.worker2task(thread_idx()):
                sb += AssignStmt(buf[r], expf(buf[r] - rv))
            # calculate sum(exp(v-max))
            sb += AssignStmt(rv, 0.0)
            for r, in block_layout.worker2task(thread_idx()):
                sb += AssignStmt(rv, rv + if_then_else(r < reduce_extent, buf[r], 0.0))
            sb += self.warp_reduce(rv, lambda a, b: a + b)
            # calculate exp(v-max) / sum(exp(vv-max))
            for r, in block_layout.worker2task(thread_idx()):
                with sb.if_then(r < reduce_extent):
                    sb += BufferStoreStmt(y, other_indices[:axis] + (r,) + other_indices[axis:], buf[r] / rv)
            fb.set_body(sb.finish())
        func = fb.get()
        return IRModule(funcs={func.name: func}, task=task)

    def warp_reduce(self, v, op) -> Stmt:
        sb = StmtBuilder()
        with sb.let('mask', active_mask()) as mask:
            for delta in [16, 8, 4, 2, 1]:
                sb += AssignStmt(v, op(v, shfl_down_sync(mask, v, delta=delta)))
            sb += AssignStmt(v, shfl_sync(mask, v, src_lane=0))
        return sb.finish()

