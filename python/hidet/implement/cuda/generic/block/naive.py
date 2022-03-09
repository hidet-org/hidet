from typing import Mapping, List, Tuple, Any
from contextlib import ExitStack

from hidet.ir.node import Node
from hidet.ir.type import scalar_type, tensor_type, TensorType
from hidet.ir.expr import Expr, Var, Call, convert, Constant, TensorElement, And, var
from hidet.ir.stmt import Stmt, IfStmt, EvaluateStmt, concat_stmts, SeqStmt, BufferStoreStmt, ForStmt
from hidet.ir.task import Task, Grid, ThreadBlock, Thread
from hidet.ir.func import IRModule, Function, FunctionGroup
from hidet.ir.dialects.compute import TensorInput, ScalarInput, TensorCompute, compute, scalar_input
from hidet.ir.dialects.pattern import TaskPattern, TensorComputePattern, ScalarExprPattern, OptionalPattern
from hidet.ir.dialects.lowlevel import VoidType, Address
from hidet.ir.functors import rewrite, infer_type, collect, simplify
from hidet.implement.implementer import Implementer, implement, register_impl, NotSupportedError
from hidet.implement.common import expand_loop
from hidet.ir.primitives import thread_idx
from hidet.ir.layout import TaskLayout, row_major_layout, RowMajorLayout, full_layout
from hidet.ir.builders import FunctionBuilder, StmtBuilder


def reduce_product(lst: List[Expr]):
    if len(lst) == 0:
        return convert(1)
    s = lst[0]
    for v in lst[1:]:
        s = s * v
    return s


@register_impl('cuda_block_naive_implementer')
class CudaBlockNaiveImplementer(Implementer):
    def __init__(self):
        self.block_size = Constant(None, dtype=scalar_type('int32'))
        self.computation = TensorComputePattern(rank=None, allow_dynamic_axis=False)
        self.task_layout = OptionalPattern(TaskLayout())
        self.pattern = TaskPattern(
            compute_pattern=self.computation,
            required_params=[],
            required_param_types=[],
            allow_tensor_extra_params=True,
            worker=ThreadBlock(block_dim=self.block_size, task_layout=self.task_layout)
        )

    def priority(self) -> int:
        return 0

    def task_pattern(self) -> TaskPattern:
        return self.pattern

    def implement(self, task: Task, match: Mapping[Node, Any]) -> IRModule:
        block_size = int(match[self.block_size])
        computation: TensorCompute = match[self.computation]
        if match[self.task_layout]:
            task_layout: TaskLayout = match[self.task_layout]
            ir_module = IRModule(task=task)
            ir_module.include(self.implement_for_given_layout(task, match, task_layout))
            return ir_module
        else:
            atom_layouts = RowMajorLayout.get_layouts(num_workers=block_size, rank=len(computation.shape))
            ir_module = IRModule(task=task)
            for atom_layout in atom_layouts:
                try:
                    ir_module.include(self.implement_for_atom_layout(task, match, atom_layout))
                except NotSupportedError:
                    pass
                else:  # return the first successful implementation
                    break
            return ir_module

    def implement_for_given_layout(self, task: Task, match: Mapping[Node, Any], given_layout: TaskLayout):
        computation: TensorCompute = match[self.computation]
        task_shape = [int(v) for v in computation.shape]
        block_size = int(match[self.block_size])
        assert all(a == b for a, b in zip(task_shape, given_layout.task_shape))
        assert given_layout.num_workers == int(block_size)
        with FunctionBuilder(task.name, attrs={'worker': task.worker}) as fb:
            # params
            param_vars = [Var(param.name, param_type) for param, param_type in zip(task.params, task.params_type)]
            param2var = {p: v for p, v in zip(task.params, param_vars)}
            fb.extend_params(param_vars)
            # body
            sb = StmtBuilder()
            for task_index in given_layout.worker2task(thread_idx()):
                rmap = {axis: task_index for axis, task_index in zip(computation.axes, task_index)}
                value = rewrite(computation.value, rmap)
                with ExitStack() as stack:
                    if computation.predicate is not None:
                        stack.enter_context(sb.if_then(rewrite(computation.predicate, rewrite_map={**param2var, **rmap})))
                    stmt, scalar_value, new_var_map = expand_loop(value, input_map=param2var)
                    sb += stmt
                    fb.extend_local_vars(new_var_map.values())  # the local variables required to expand the value.
                    if computation.accumulate == 'sum':
                        scalar_value = TensorElement(param2var[computation], task_index) + scalar_value
                    elif computation.accumulate is None:
                        pass
                    else:
                        raise NotImplementedError()
                    sb += BufferStoreStmt(param2var[computation], task_index, scalar_value)
            fb.set_body(sb.finish())
        return IRModule(funcs={task.name: fb.get()}, task=task)

    def implement_for_atom_layout(self, task: Task, match: Mapping[Node, Node], atom_layout: TaskLayout) -> IRModule:
        block_size = int(match[self.block_size])
        computation: TensorCompute = match[self.computation]
        task_shape: List[int] = [int(v) for v in computation.shape]
        atom_shape: List[int] = atom_layout.task_shape
        rank = len(task_shape)
        self.check(block_size == atom_layout.num_workers)
        self.check(a % b == 0 for a, b in zip(task_shape, atom_layout.task_shape))
        with FunctionBuilder(task.name, attrs={'worker': task.worker}) as fb:
            # params
            param_vars = [Var(param.name, param_type) for param, param_type in zip(task.params, task.params_type)]
            param2var = {p: v for p, v in zip(task.params, param_vars)}
            fb.extend_params(param_vars)
            # body
            sb = StmtBuilder()
            with ExitStack() as stack:
                loop_vars = []
                atom_vars = []
                assert len(atom_layout.worker2task(thread_idx())) == 1
                thread_task = atom_layout.worker2task(thread_idx())[0]
                for i in range(rank):
                    atom_var = stack.enter_context(sb.let(f'i{i}', thread_task[i]))
                    atom_vars.append(atom_var)
                for i in range(rank):
                    loop_var = stack.enter_context(sb.for_loop(f'o{i}', (task_shape[i] + atom_shape[i] - 1) // atom_shape[i]))
                    loop_vars.append(loop_var)
                cond = convert(True)
                task_indices = []
                for loop_var, atom_var, loop_dim, bound in zip(loop_vars, atom_vars, atom_shape, task_shape):
                    task_index = loop_var * loop_dim + atom_var
                    task_indices.append(task_index)
                    cond = And(cond,  task_index < bound)
                with sb.if_then(cond):
                    rmap = {axis: task_index for axis, task_index in zip(computation.axes, task_indices)}
                    value = rewrite(computation.value, rmap)
                    stmt, scalar_value, new_var_map = expand_loop(value, input_map=param2var)
                    sb += stmt
                    sb += BufferStoreStmt(param2var[computation], task_indices, scalar_value)
                    fb.extend_local_vars(new_var_map.values())  # the local variables required to expand the value.
            fb.set_body(sb.finish())
        return IRModule(funcs={task.name: fb.get()}, task=task)

