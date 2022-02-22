from typing import Mapping, Callable, Any, Tuple

from hidet.ir import TensorInput, ScalarInput, ReduceCompute, TensorCompute
from hidet.ir.builders import StmtBuilder
from hidet.ir.dialects.compute import compute
from hidet.ir.dialects.lowlevel import Cast, Dereference
from hidet.ir.expr import *
from hidet.ir.func import IRModule
from hidet.ir.task import Worker, ThreadBlock, Warp
from hidet.ir.functors import ExprFunctor, infer_type, rewrite, ExprRewriter
from hidet.ir.stmt import ForStmt, BufferStoreStmt, AssignStmt, SeqStmt
from hidet.ir.builders import TaskBuilder


def transfer_task(name: str, src_type: TensorType, dst_type: TensorType, worker: Worker, parent_module: IRModule) -> TaskBuilder:
    with TaskBuilder(name, worker, parent_module) as tb:
        src = TensorInput('src', dtype=src_type.scalar_type)
        dst = compute('dst', shape=dst_type.shape, fcompute=lambda *args: src.__getitem__(args))
        tb.set_computation(dst)
        tb.append_param(src, src_type)
        tb.append_param(dst, dst_type)
    return tb


def transfer_predicated_task(name: str, cond: Callable, src_type: TensorType, dst_type: TensorType, worker: Worker, parent_module: IRModule, default_value=None) -> TaskBuilder:
    if default_value is None:
        default_value = convert(0.0)

    with TaskBuilder(name, worker, parent_module) as tb:
        src = TensorInput('src', dtype=src_type.scalar_type)
        dst = compute('dst', shape=dst_type.shape, fcompute=lambda *args: if_then_else(cond(*args), src.__getitem__(args), default_value))
        tb.set_computation(dst)
        tb.append_param(src, src_type)
        tb.append_param(dst, dst_type)
    return tb


def predicated_transfer_task(name: str, cond: Callable, src_type: TensorType, dst_type: TensorType, worker: Worker, parent_module: IRModule, aux_params=(), aux_params_type=()) -> TaskBuilder:
    with TaskBuilder(name, worker, parent_module) as tb:
        src = TensorInput('src', dtype=src_type.scalar_type)
        dst = compute(name='dst',
                      shape=dst_type.shape,
                      fcompute=lambda *args: src.__getitem__(args),
                      predicate=lambda *args: cond(*args))
        tb.set_computation(dst)
        tb.append_param(src, src_type)
        tb.append_param(dst, dst_type)
        for aux_param, aux_param_type in zip(aux_params, aux_params_type):
            tb.append_param(aux_param, aux_param_type)
    return tb


def transfer_bounded_task(name: str, src_type: TensorType, dst_type: TensorType, worker: Worker, parent_module: IRModule, default_value=None) -> TaskBuilder:
    if default_value is None:
        default_value = convert(0.0)

    shape = dst_type.shape
    with TaskBuilder(name, worker, parent_module) as tb:
        rank = len(dst_type.shape)
        bound_params = [ScalarInput('d', 'int32') for _ in range(rank)]
        bound_params_type = [ScalarType('int32') for _ in range(rank)]
        src = TensorInput('src', dtype=src_type.scalar_type)
        fcompute = lambda *args: if_then_else(And.join(*[args[i] < bound_params[i] for i in range(len(shape))]), src.__getitem__(args), default_value)
        dst = compute('dst', shape=shape, fcompute=fcompute)
        tb.set_computation(dst)
        tb.append_param(src, src_type)
        tb.append_param(dst, dst_type)
        for bound_param, bound_param_type in zip(bound_params, bound_params_type):
            tb.append_param(bound_param, bound_param_type)
    return tb


def bounded_transfer_task(name: str, src_type: TensorType, dst_type: TensorType, worker: Worker, parent_module: IRModule) -> TaskBuilder:
    with TaskBuilder(name, worker, parent_module) as tb:
        rank = len(dst_type.shape)
        bound_params = [ScalarInput('d', 'int32') for _ in range(rank)]
        bound_params_type = [ScalarType('int32') for _ in range(rank)]
        src = TensorInput('src', dtype=src_type.scalar_type)
        dst = compute(name='dst',
                      shape=dst_type.shape,
                      fcompute=lambda *args: src.__getitem__(args),
                      predicate=lambda *args: And.join(*[a < b for a, b in zip(args, bound_params)]))
        tb.set_computation(dst)
        tb.append_param(src, src_type)
        tb.append_param(dst, dst_type)
        for bound_param, bound_param_type in zip(bound_params, bound_params_type):
            tb.append_param(bound_param, bound_param_type)
    return tb


def init_task(name: str, dst_type: TensorType, init_value: Union[Expr, PyScalar], worker: Worker, parent_module: IRModule) -> TaskBuilder:
    with TaskBuilder(name, worker, parent_module) as tb:
        dst = compute('dst', shape=dst_type.shape, fcompute=lambda *args: convert(init_value))
        tb.set_computation(dst)
        tb.append_param(dst, dst_type)
    return tb


class LoopExpander(ExprRewriter):
    def __init__(self, input_map):
        super().__init__()
        self.sb = StmtBuilder()
        self.input_map = input_map
        self.new_buffer_map = {}

    def expand(self, e):
        value = self.visit(e)
        return self.sb.finish(), value, self.new_buffer_map

    def visit_TensorInput(self, e: TensorInput):
        return self.input_map[e]

    def visit_ScalarInput(self, e: ScalarInput):
        return self.input_map[e]

    def visit_TensorCompute(self, e: TensorCompute):
        # declare output buffer when needed
        if e in self.input_map:
            buf = self.input_map[e]
        else:
            buf = tensor_var(e.name, e.shape, dtype=infer_type(e.value))
            self.new_buffer_map[e] = buf

        # tensor compute loops
        for i in range(len(e.shape)):
            self.sb.enter_body(ForStmt(e.axes[i], e.shape[i]))

        # at the inner-most loop body
        expr = self.visit(e.value)
        if e.accumulate:
            if e.accumulate == 'sum':
                expr = buf.__getitem__(tuple(e.axes)) + expr
            else:
                raise NotImplementedError()
        self.sb.append(BufferStoreStmt(buf, e.axes, expr))

        # exit loop scope
        for i in range(len(e.shape)):
            self.sb.exit_body()

        return buf

    def visit_ReduceCompute(self, e: ReduceCompute):
        extent = e.shape[0]
        if isinstance(extent, Constant) and extent.value == 1:
            value_expr = self.visit(e.value)
            acc = rewrite(value_expr, {e.axis: convert(0)})
        else:
            # declare accumulator
            acc = scalar_var(e.name, infer_type(e.value))
            self.new_buffer_map[e] = acc

            # init accumulator
            self.sb += AssignStmt(acc, e.init_const())

            # reduction loop
            assert len(e.shape) == 1
            with self.sb.for_loop(e.axis, e.shape[0]):
                expr = self.visit(e.value)
                self.sb += AssignStmt(acc, e.combine(acc, expr))

        # if e is in the input buffer, we should write it back
        if e in self.input_map:
            input_var = self.input_map[e]
            self.sb += AssignStmt(input_var, acc)

        return acc


def expand_loop(expr: Expr, input_map: Mapping[Union[ScalarInput, TensorInput, Expr], Var]):
    """
    Generate statements to calculate the expression.

    The expression may contain TensorCompute and ReduceCompute sub-expressions.
    After expand, the stmt will not have ScalarInput, TensorInput, TensorCompute and ReduceCompute anymore.

    The returned new_buffer_map is a mapping from ReduceCompute and TensorCompute sub-expressions to
    new allocated buffers used to finish the computation.

    For example, the following expr:
    compute([3, 3], (i, j) -> reduce_sum(A[i, k] * B[k, j], axis=k)) where k = axis(3)
    will be expanded to
    for i in range(3):
        for j in range(3):
            s = 0
            for k in range(3):
                s += A[i, k] * B[k, j]
            C[i, j] = s

    If C is in input_map, then the mapped var is used directly. Otherwise, a new tensor var is created to store the results
    and returned in new_buffer_map. We only reuse tensor in input_map.
    """
    expander = LoopExpander(input_map)
    stmt, value, new_buffer_map = expander.expand(expr)
    return stmt, value, new_buffer_map
