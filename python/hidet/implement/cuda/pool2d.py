from typing import Mapping, Any, List

from hidet.implement.common import VirtualTensor, pattern2matched
from hidet.implement.implementer import register_impl, Implementer
from hidet.ir import IRModule
from hidet.ir.builders import FunctionBuilder, StmtBuilder
from hidet.ir.dialects.compute import TensorCompute, ReduceCompute
from hidet.ir.dialects.pattern import TaskPattern, any_const_ints, any_scalar_expr, int_vars, StringPattern
from hidet.ir.expr import Var, scalar_var, convert
from hidet.ir.functors import rewrite
from hidet.ir.layout import TaskLayout
from hidet.ir.node import Node
from hidet.ir.primitives import block_idx, thread_idx, cuda_max
from hidet.ir.stmt import BufferStoreStmt, AssignStmt
from hidet.ir.task import Task, Grid
from hidet.utils.info import float_type_min_value


class Pattern:
    def __init__(self):
        self.n, self.c, self.p, self.q, self.rx, self.ry = any_const_ints(6)
        self.an, self.ac, self.ap, self.aq, self.arx, self.ary = int_vars(['an', 'ac', 'ap', 'aq', 'arx', 'ary'])
        self.x_expr = any_scalar_expr()  # can only use [n, rc, p, q, rx, ry]
        self.reduce_type = StringPattern()
        self.y = TensorCompute(
            name='out',
            shape=[self.n, self.c, self.p, self.q],
            axes=[self.an, self.ac, self.ap, self.aq],
            value=ReduceCompute(
                shape=[self.rx, self.ry],
                axes=[self.arx, self.ary],
                value=self.x_expr,
                reduce_type=self.reduce_type,
                data_type=None,
            ),
            data_type=None
        )
        self.task_pattern = TaskPattern(
            compute_pattern=self.y,
            required_params=[self.y],
            worker=Grid()
        )


@register_impl('cuda_grid_pool2d_implementer')
class CudaGridPool2dImplementer(Implementer):
    def __init__(self):
        self.pattern = Pattern()

    def priority(self) -> int:
        return 1

    def task_pattern(self) -> TaskPattern:
        return self.pattern.task_pattern

    def implement(self, task: Task, match: Mapping[Node, Any]) -> IRModule:
        md = pattern2matched(self.pattern, match)
        grid_layout = TaskLayout.row_major([md.n, md.c, md.p, md.q])
        op_init = {
            'max': float_type_min_value(),
            'avg': 0.0
        }
        op_dict = {
            'max': cuda_max,
            'avg': lambda a, b: a + b
        }
        reduce_init_value = op_init[md.reduce_type]
        reduce_op = op_dict[md.reduce_type]

        block_size = 512
        grid_size = (grid_layout.num_workers + block_size - 1) // block_size
        with FunctionBuilder(task.name + '_grid', worker=Grid(grid_dim=grid_size, block_dim=block_size)) as fb:
            # params
            params: List[Var] = [Var(param.name, param_type) for param, param_type in zip(task.params, task.param_types())]
            fb.extend_params(params)
            param_map = {task_param: func_param for task_param, func_param in zip(task.params, params)}
            x = VirtualTensor(lambda n, c, p, q, rx, ry: rewrite(md.x_expr, {md.an: n, md.ac: c, md.ap: p, md.aq: q, md.arx: rx, md.ary: ry, **param_map}))
            y = param_map[md.y]

            # local vars
            reduce_value = scalar_var('reduce_value', 'float32')
            fb.extend_local_vars([reduce_value])

            # body
            sb = StmtBuilder()
            with sb.let('wid', thread_idx() + block_idx() * block_size) as wid:
                n, c, p, q = grid_layout(wid)[0]
                with sb.if_then(wid < grid_layout.num_workers):
                    sb += AssignStmt(reduce_value, reduce_init_value)
                    with sb.for_loop('rx', md.rx) as rx:
                        with sb.for_loop('ry', md.ry) as ry:
                            sb += AssignStmt(reduce_value, reduce_op(reduce_value, x[n, c, p, q, rx, ry]))
                    if md.reduce_type == 'avg':
                        sb += AssignStmt(reduce_value, reduce_value / convert(float(md.rx * md.ry)))
                    sb += BufferStoreStmt(y, [n, c, p, q], reduce_value)
            fb.set_body(sb.finish())
        func = fb.get()
        return IRModule(funcs={func.name: func}, task=task)

