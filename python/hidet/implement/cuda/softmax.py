from typing import Mapping, Any, List
import sys

from hidet.implement.implementer import Implementer, register_impl
from hidet.ir import IRModule
from hidet.ir.type import tensor_type, TensorType, scalar_type
from hidet.ir.layout import DataLayout, TaskLayout
from hidet.ir.expr import convert, Constant, Var, scalar_var, if_then_else, tensor_var
from hidet.ir.stmt import AssignStmt, Stmt, BufferStoreStmt
from hidet.ir.dialects.compute import tensor_input, compute, reduce, TensorInput
from hidet.ir.dialects.pattern import TaskPattern, any_const_int
from hidet.ir.primitives import expf, block_idx, thread_idx, cuda_max, active_mask, shfl_down_sync, shfl_sync
from hidet.ir.node import Node
from hidet.ir.task import Task, Grid
from hidet.ir.functors import rewrite
from hidet.ir.builders import FunctionBuilder, StmtBuilder


def pattern2matched(pattern, match):
    matched = type(pattern)()
    for name in matched.__dict__:
        v = match[pattern.__dict__[name]]
        if isinstance(v, Constant):
            v = v.value
        if isinstance(v, (list, tuple)):
            v = (vv.value if isinstance(vv, Constant) else vv for vv in v)
        matched.__dict__[name] = v
    return matched


class Pattern:
    def __init__(self):
        shape = (any_const_int(), any_const_int(), any_const_int(), any_const_int())
        m, n, p, q = shape
        x = TensorInput('x', TensorType(shape=(m, n, p, q)))
        mx = compute(
            name='mx',
            shape=[m, p, q],
            fcompute=lambda i, r, s: reduce(
                shape=[n],
                fcompute=lambda j: x[i, j, r, s],
                reduce_type='max'
            )
        )
        e = compute(
            name='e',
            shape=[m, n, p, q],
            fcompute=lambda i, j, r, s: expf(x[i, j, r, s] - mx[i, r, s])
        )
        se = compute(
            name='se',
            shape=[m, p, q],
            fcompute=lambda i, r, s: reduce(
                shape=[n],
                fcompute=lambda j: e[i, j, r, s],
                reduce_type='sum'
            )
        )
        y = compute(
            name='y',
            shape=[m, n, p, q],
            fcompute=lambda i, j, r, s: e[i, j, r, s] / se[i, r, s]
        )
        task_pattern = TaskPattern(
            compute_pattern=y,
            required_params=[x, y],
            required_param_types=[
                tensor_type(scope='global', dtype='float32', shape=[None, None, None, None]),
                tensor_type(scope='global', dtype='float32', shape=[None, None, None, None])
            ],
            allow_extra_params=False,
            worker=Grid()
        )
        self.shape = shape
        self.x = x
        self.y = y
        self.task_pattern = task_pattern


@register_impl('cuda_grid_softmax_implementer')
class CudaGridSoftmaxImplementer(Implementer):
    def __init__(self):
        super().__init__()
        self.pattern = Pattern()

    def priority(self) -> int:
        return 1

    def task_pattern(self) -> TaskPattern:
        return self.pattern.task_pattern

    def implement(self, task: Task, match: Mapping[Node, Any]) -> IRModule:
        d = pattern2matched(self.pattern, match)
        n, c, h, w = d.shape
        warp_size = 32
        outer = (c + warp_size) // warp_size
        grid_layout = TaskLayout.row_major([n, h, w])
        block_layout = TaskLayout.full_layout([outer]) * TaskLayout.row_major([warp_size])
        with FunctionBuilder(task.name + '_grid', worker=Grid(grid_dim=grid_layout.num_workers, block_dim=block_layout.num_workers), label='softmax') as fb:
            # params
            params: List[Var] = [Var(param.name, param_type) for param, param_type in zip(task.params, task.param_types())]
            fb.extend_params(params)
            x, y = params

            # local variables
            buf = tensor_var('buf', shape=[outer], scope='register', dtype='float32', layout=DataLayout.row_major([outer]))
            rv = scalar_var('rv', 'float32')
            fb.extend_local_vars([rv, buf])

            # body
            sb = StmtBuilder()
            # get the max value along c dimension
            sb += AssignStmt(rv, sys.float_info.min)
            i, r, s = grid_layout.worker2task(block_idx())[0]
            for j, in block_layout.worker2task(thread_idx()):
                with sb.if_then(j < c):
                    sb += BufferStoreStmt(buf, [j], x[i, j, r, s])
                    sb += AssignStmt(rv, cuda_max(rv, buf[j]))
            sb += self.warp_reduce(rv, cuda_max)
            # calculate exp(v-max)
            for j, in block_layout.worker2task(thread_idx()):
                sb += AssignStmt(buf[j], expf(buf[j] - rv))
            # calculate sum(exp(v-max))
            sb += AssignStmt(rv, 0.0)
            for j, in block_layout.worker2task(thread_idx()):
                sb += AssignStmt(rv, rv + if_then_else(j < c, buf[j], 0.0))
            sb += self.warp_reduce(rv, lambda a, b: a + b)
            # calculate exp(v-max) / sum(exp(vv-max))
            for j, in block_layout.worker2task(thread_idx()):
                with sb.if_then(j < c):
                    sb += BufferStoreStmt(y, [i, j, r, s], buf[j] / rv)
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






