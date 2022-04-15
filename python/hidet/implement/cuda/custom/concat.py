import contextlib
import sys
from typing import Mapping, Any, List, Tuple, Union

from hidet.implement.implementer import register_impl, Implementer, NotSupportedError, Schedule
from hidet.ir import IRModule
from hidet.ir.builders import FunctionBuilder, StmtBuilder
from hidet.ir.dialects.compute import TensorCompute, ReduceCompute, compute, CustomCompute, TensorInput
from hidet.ir.dialects.pattern import TaskPattern, any_const_int, any_const_ints, any_scalar_expr, int_vars, StringPattern, AnyExpr, TensorComputePattern
from hidet.ir.expr import Var, Constant, And, if_then_else, Equal, scalar_var
from hidet.ir.stmt import BufferStoreStmt, IfStmt, AssignStmt, ReturnStmt
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
        self.axis = any_const_int()
        self.compute = CustomCompute(
            name=None,
            identifier='concat',
            params=None,
            data_type=None,
            attributes={'axis': self.axis}
        )
        self.task_pattern = TaskPattern(
            compute_pattern=self.compute,
            worker=Grid()
        )


@register_impl('cuda_static_concat_implementer')
class CudaStaticConcatImplementer(Implementer):
    """
    a, b, x, c, d
    ctaid.y <=> x
    ctaid.x
    tid.x 512
    """
    def __init__(self):
        self.pattern = Pattern()

    def priority(self) -> int:
        return 1

    def task_pattern(self) -> TaskPattern:
        return self.pattern.task_pattern

    def implement(self, task: Task, match: Mapping[Node, Any]) -> IRModule:
        p = pattern2matched(self.pattern, match)
        computation: CustomCompute = p.compute
        inputs: List[Union[TensorInput, TensorCompute]] = [p for p in computation.params if isinstance(p, (TensorInput, TensorCompute))]
        assert len(inputs) == len(computation.params)
        axis: int = p.axis
        shapes = [t.const_shape() for t in inputs]
        other_shape = list(v for i, v in enumerate(shapes[0]) if i != axis)
        concat_sizes: List[int] = [shape[axis] for shape in shapes]
        other_size: int = prod(other_shape)
        block_dim_x = 500
        grid_dim_x = (other_size + block_dim_x - 1) // block_dim_x
        grid_dim_y = sum(concat_sizes)
        with FunctionBuilder(name=task.name + '_grid', worker=Grid(grid_dim=(grid_dim_x, grid_dim_y), block_dim=block_dim_x)) as fb:
            # params
            params = [Var(param.name, param.data_type) for param in task.params]
            param_map = {param: var for param, var in zip(task.params, params)}
            fb.extend_params(params)
            # body
            sb = StmtBuilder()
            prefix_sum = 0
            input_tensors = params[:-1]
            output_tensor = params[-1]
            for i, tensor in enumerate(input_tensors):
                with sb.if_then(block_idx('y') < prefix_sum + concat_sizes[i]):
                    task_layout = TaskLayout.row_major(other_shape)
                    with sb.let('wid', thread_idx() + block_idx() * block_dim_x) as wid:
                        with sb.if_then(wid < other_size):
                            indices = list(task_layout(wid)[0])
                            input_indices = indices[:axis] + [block_idx('y') - prefix_sum] + indices[axis:]
                            output_indices = indices[:axis] + [block_idx('y')] + indices[axis:]
                            sb += BufferStoreStmt(output_tensor, output_indices, tensor[input_indices])
                    sb += ReturnStmt()
                prefix_sum += concat_sizes[i]
            fb.set_body(sb.finish())
        func = fb.get()
        return IRModule(funcs={func.name: func}, task=task)

