from __future__ import annotations
from typing import Mapping

from hidet.ir.dialects.compute import TensorNode, ScalarNode
from hidet.ir.builders import StmtBuilder
from hidet.ir.expr import *
from hidet.ir.functors import infer_type, ExprRewriter, rewrite
from hidet.ir.stmt import ForStmt, BufferStoreStmt, AssignStmt
from hidet.ir.task import Task
from hidet.utils import prod


class NotSupportedError(Exception):
    pass


class Schedule:
    def keys(self) -> List[Tuple[str, Union[int, float, str]]]:
        raise NotImplementedError()

    def derived_keys(self) -> List[Tuple[str, Union[int, float, str]]]:
        raise NotImplementedError()


class LoopExpander(ExprRewriter):
    def __init__(self, input_map):
        super().__init__()
        self.sb = StmtBuilder()
        self.input_map = input_map
        self.new_buffer_map = {}

    def expand(self, e):
        value = self.visit(e)
        return self.sb.finish(), value, self.new_buffer_map

    # def visit_TensorInput(self, e: TensorNode):
    #     return self.input_map[e]
    #
    # def visit_ScalarInput(self, e: ScalarNode):
    #     return self.input_map[e]
    #

    def visit_TensorNode(self, e: TensorNode):
        if e.grid_compute is None:
            # input tensor
            return self.input_map[e]
        grid_compute = e.grid_compute
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

        return buf

    def visit_ScalarNode(self, e: ScalarNode):
        if e.reduce_compute is None:
            # input scalar
            return self.input_map[e]

        rc = e.reduce_compute
        shape, axes, value = rc.shape, rc.axes, rc.value
        # declare accumulator
        acc = scalar_var(e.name, infer_type(value))
        self.new_buffer_map[e] = acc

        # init accumulator
        self.sb += AssignStmt(acc, rc.init_const(e.data_type.name))

        # reduction loops
        for i in range(len(shape)):
            self.sb.enter_body(ForStmt(axes[i], shape[i]))

        # at the innermost loop body
        expr = self.visit(value)
        self.sb += AssignStmt(acc, rc.combine(acc, expr))

        # exit loop scope
        for i in range(len(shape)):
            self.sb.exit_body()

        # finalize
        acc = rc.finalize(acc, prod(shape))

        # if e is in the input buffer, we should write it back
        if e in self.input_map:
            input_var = self.input_map[e]
            self.sb += AssignStmt(input_var, acc)

        return acc


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


class VirtualTensor:
    """
    A virtual tensor map index to a value
    VirtualTensor can be used to abstract an expression to a tensor.
    Support indexing and slicing.

    For example, considering this expression: 0 <= i && i < 32 ? A[i] : 0.0, we can construct a
    virtual tensor A = VirtualTensor(fmap=lambda i: 0<=i && i<32 ? A[i] : 0.0);
    Then we can access A[i] and slice A[1:].
    """

    def __init__(self, fmap):
        self.fmap = fmap

    def __getitem__(self, item):
        if not isinstance(item, (list, tuple)):
            item = [item]
        if any(isinstance(v, slice) for v in item):
            starts = []
            indices = []
            for v in item:
                if isinstance(v, slice):
                    starts.append(v.start if v.start else 0)
                    indices.append(None)
                else:
                    starts.append(None)
                    indices.append(v)

            def fmap(*slice_indices):
                assert len(indices) == len([v for v in starts if v is not None])
                orig_indices = []
                cur = 0
                for i in range(len(starts)):
                    if starts[i] is not None:
                        orig_indices.append(slice_indices[cur] + starts[i])
                        cur += 1
                    else:
                        orig_indices.append(indices[i])
                return self.__getitem__(orig_indices)

            return VirtualTensor(fmap)
        else:
            return self.fmap(*item)

    @staticmethod
    def from_indexed_value(indices: Sequence[Var], value: Expr) -> VirtualTensor:
        def fmap(*actual_indices):
            if len(actual_indices) != len(indices):
                raise ValueError('Expect {} number of indices, got {}.'.format(len(indices), len(actual_indices)))
            return rewrite(value, {a: b for a, b in zip(indices, actual_indices)})

        return VirtualTensor(fmap)


def params_from_task(task: Task) -> List[Var]:
    return [Var(param.name, param.data_type) for param in task.parameters]


def inputs_from_task(task: Task, params: List[Var]) -> List[Union[VirtualTensor, Var]]:
    inputs = []
    param2var = {param: var for param, var in zip(task.parameters, params)}
    for input in task.inputs:
        if input in task.prologues:
            prologue = task.prologues[input]
            value = rewrite(prologue.value, param2var)
            inputs.append(VirtualTensor.from_indexed_value(prologue.indices, value))
        else:
            assert input in param2var
            inputs.append(param2var[input])
    return inputs


def outputs_from_task(task: Task, params: List[Var]) -> List[Var]:
    outputs = []
    param2var = {param: var for param, var in zip(task.parameters, params)}
    for output in task.outputs:
        assert output in param2var
        outputs.append(param2var[output])
    return outputs


def write_output(buf: Var, indices: List[Var], value: Expr, task: Task, params: List[Var]) -> BufferStoreStmt:
    param2var = {param: var for param, var in zip(task.parameters, params)}
    var2param = {var: param for param, var in zip(task.parameters, params)}
    param = var2param[buf]
    if param in task.epilogues:
        epilogue = task.epilogues[param]
        rmap = param2var
        rmap.update({a: b for a, b in zip(epilogue.indices, indices)})
        value = rewrite(epilogue.value, rmap)
        return BufferStoreStmt(buf, indices, value)
    else:
        return BufferStoreStmt(buf, indices, value)
