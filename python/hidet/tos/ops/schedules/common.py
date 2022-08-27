from __future__ import annotations
from typing import Mapping

from hidet.ir.dialects.compute import TensorNode, ScalarNode, GridCompute, ArgReduceCompute, ReduceCompute
from hidet.ir.builders import StmtBuilder
from hidet.ir.expr import *
from hidet.ir.functors import infer_type, ExprRewriter, rewrite
from hidet.ir.stmt import ForStmt, BufferStoreStmt, AssignStmt
from hidet.ir.task import Task
from hidet.utils import prod


class NotSupportedError(Exception):
    def __init__(self, obj: object, msg: str = ""):
        self.obj = obj
        self.msg = msg


class Schedule:
    def __str__(self):
        items = []
        for name, value in self.keys():
            items.append('{}: {}'.format(name, value))
        schedule_type = self.__class__.__name__
        schedule_keys = ', '.join(items)
        return '{}({})'.format(schedule_type, schedule_keys)

    def __repr__(self):
        return str(self)

    def keys(self) -> List[Tuple[str, Union[int, float, str]]]:
        raise NotImplementedError()

    def derived_keys(self) -> List[Tuple[str, Union[int, float, str]]]:
        raise NotImplementedError()

    def check(self, cond, msg=''):
        if not cond:
            raise NotSupportedError(self, msg)


class LoopExpander(ExprRewriter):
    def __init__(self, input_map):
        super().__init__()
        self.sb = StmtBuilder()
        self.input_map = input_map
        self.new_buffer_map = {}

    def expand(self, e):
        value = self.visit(e)
        return self.sb.finish(), value, self.new_buffer_map

    def visit_TensorNode(self, e: TensorNode):
        if e.tensor_compute is None:
            # input tensor
            return self.input_map[e]
        tc = e.tensor_compute
        if isinstance(tc, GridCompute):
            grid_compute = e.tensor_compute
            # declare output buffer when needed
            if e in self.input_map:
                buf = self.input_map[e]
            else:
                buf = Var(e.name, e.data_type)
                self.new_buffer_map[e] = buf

            shape, axes, value = grid_compute.shape, grid_compute.axes, grid_compute.value
            # tensor compute loops
            for i in range(len(shape)):
                self.sb.enter_body(ForStmt(axes[i], shape[i]))

            # at the innermost loop body
            expr = self.visit(grid_compute.value)
            self.sb.append(BufferStoreStmt(buf, axes, expr))

            # exit loop scope
            for i in range(len(shape)):
                self.sb.exit_body()
        elif isinstance(tc, ArgReduceCompute):
            raise NotImplementedError('Compute pattern {}'.format(type(tc).__name__))
        else:
            raise NotImplementedError('Compute pattern {}'.format(type(tc).__name__))
        return buf

    def visit_ScalarNode(self, e: ScalarNode):
        if e.scalar_compute is None:
            # input scalar
            return self.input_map[e]

        sc = e.scalar_compute
        if isinstance(sc, ReduceCompute):
            rc = e.scalar_compute
            shape, axes, value = rc.shape, rc.axes, rc.value
            # declare accumulator
            acc = scalar_var(e.name, infer_type(value))
            self.new_buffer_map[e] = acc

            # init accumulator
            self.sb += AssignStmt(acc, rc.reduce_operation.initial_value(e.data_type.name))

            # reduction loops
            for i in range(len(shape)):
                self.sb.enter_body(ForStmt(axes[i], shape[i]))

            # at the innermost loop body
            expr = self.visit(value)
            self.sb += AssignStmt(acc, rc.reduce_operation.combine(acc, expr))

            # exit loop scope
            for i in range(len(shape)):
                self.sb.exit_body()

            # finalize
            acc = rc.reduce_operation.finalize(acc, prod(shape))

            # if e is in the input buffer, we should write it back
            if e in self.input_map:
                input_var = self.input_map[e]
                self.sb += AssignStmt(input_var, acc)

            return acc
        else:
            raise NotImplementedError('Compute pattern {}'.format(type(sc).__name__))


def expand_loop(expr: Expr, input_map: Mapping[Union[ScalarNode, TensorNode], Var]):
    """
    Generate statements to calculate the expression.

    The expression may contain TensorCompute and ReduceCompute sub-expressions.
    After expand, the stmt will not have ScalarInput, TensorInput, TensorCompute and ReduceCompute anymore.

    The returned new_buffer_map is a mapping from ReduceCompute and TensorCompute sub-expressions to
    new allocated buffers used to conduct the computation.

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


# class VirtualTensor:
#     """
#     A virtual tensor map index to a value
#     VirtualTensor can be used to abstract an expression to a tensor.
#     Support indexing and slicing.
#
#     For example, considering this expression: 0 <= i && i < 32 ? A[i] : 0.0, we can construct a
#     virtual tensor A = VirtualTensor(fmap=lambda i: 0<=i && i<32 ? A[i] : 0.0);
#     Then we can access A[i] and slice A[1:].
#     """
#
#     def __init__(self, fmap):
#         self.fmap = fmap
#
#     def __getitem__(self, item):
#         if not isinstance(item, (list, tuple)):
#             item = [item]
#         if any(isinstance(v, slice) for v in item):
#             starts = []
#             indices = []
#             for v in item:
#                 if isinstance(v, slice):
#                     starts.append(v.start if v.start else 0)
#                     indices.append(None)
#                 else:
#                     starts.append(None)
#                     indices.append(v)
#
#             def fmap(*slice_indices):
#                 assert len(indices) == len([v for v in starts if v is not None])
#                 orig_indices = []
#                 cur = 0
#                 for i in range(len(starts)):
#                     if starts[i] is not None:
#                         orig_indices.append(slice_indices[cur] + starts[i])
#                         cur += 1
#                     else:
#                         orig_indices.append(indices[i])
#                 return self.__getitem__(orig_indices)
#
#             return VirtualTensor(fmap)
#         else:
#             return self.fmap(*item)
#
#     @staticmethod
#     def from_indexed_value(indices: Sequence[Var], value: Expr) -> VirtualTensor:
#         def fmap(*actual_indices):
#             if len(actual_indices) != len(indices):
#                 raise ValueError('Expect {} number of indices, got {}.'.format(len(indices), len(actual_indices)))
#             return rewrite(value, {a: b for a, b in zip(indices, actual_indices)})
#
#         return VirtualTensor(fmap)


def params_from_task(task: Task) -> List[Var]:
    return [Var(param.name, param.data_type) for param in task.inputs + task.outputs]
