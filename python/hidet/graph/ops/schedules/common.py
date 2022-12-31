# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# pylint: disable=consider-using-enumerate
from __future__ import annotations
from typing import Mapping, List, Union, Tuple

from hidet.ir.compute import TensorNode, ScalarNode, GridCompute, ArgReduceCompute, ReduceCompute
from hidet.ir.builders import StmtBuilder
from hidet.ir.expr import Expr, Var, scalar_var, convert
from hidet.ir.functors import infer_type, ExprRewriter
from hidet.ir.stmt import ForStmt, BufferStoreStmt, AssignStmt, DeclareStmt
from hidet.ir.task import Task
from hidet.utils import prod


class NotSupportedError(Exception):
    def __init__(self, obj: object, msg: str = ""):
        super().__init__()
        self.obj = obj
        self.msg = msg


class Schedule:
    def __str__(self, show_derived=False):
        items = []
        for name, value in self.keys():
            items.append('{}: {}'.format(name, value))
        for name, value in self.derived_keys():
            items.append('{}: {}'.format(name, value))
        schedule_type = self.__class__.__name__
        schedule_keys = ', '.join(items)
        return '{}({})'.format(schedule_type, schedule_keys)

    def __repr__(self):
        return self.__str__(show_derived=True)

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

    def expand(self, e):
        value = self.visit(e)
        return self.sb.finish(), value

    def visit_TensorNode(self, e: TensorNode):
        if e.tensor_compute is None:
            # input tensor
            return self.input_map[e]
        tc = e.tensor_compute
        if isinstance(tc, GridCompute):
            grid_compute = tc
            # declare output buffer when needed
            if e in self.input_map:
                buf = self.input_map[e]
            else:
                buf = Var(e.name, e.ttype)
                self.sb += DeclareStmt(buf)

            # pylint: disable=unused-variable
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
        else:
            raise NotImplementedError('Compute pattern {}'.format(type(tc).__name__))
        return buf

    def visit_ScalarNode(self, e: ScalarNode):
        if e.scalar_compute is None:
            # input scalar
            return self.input_map[e]

        sc = e.scalar_compute
        if isinstance(sc, ReduceCompute):
            shape, axes, value = sc.shape, sc.axes, sc.value
            # declare accumulator
            acc = scalar_var(e.name, infer_type(value))
            self.sb += DeclareStmt(acc, init=sc.reduce_operation.initial_value(e.dtype.name))

            # reduction loops
            for i in range(len(shape)):
                self.sb.enter_body(ForStmt(axes[i], shape[i]))

            # at the innermost loop body
            expr = self.visit(value)
            self.sb += AssignStmt(acc, sc.reduce_operation.combine(acc, expr))

            # exit loop scope
            for i in range(len(shape)):
                self.sb.exit_body()

            # finalize
            acc = sc.reduce_operation.finalize(acc, prod(shape))

            # if e is in the input buffer, we should write it back
            if e in self.input_map:
                input_var = self.input_map[e]
                self.sb += AssignStmt(input_var, acc)

            return acc
        elif isinstance(sc, ArgReduceCompute):
            extent, axis, value = sc.extent, sc.axis, sc.value
            value_dtype = infer_type(value)
            # declare index accumulator
            acc_index = scalar_var(e.name + '_idx', sc.index_dtype)
            acc_value = scalar_var(e.name + '_val', value_dtype)

            # init accumulator
            self.sb += DeclareStmt(acc_value, init=sc.reduce_operation.initial_value(value_dtype))
            self.sb += DeclareStmt(acc_index, init=convert(0))

            # reduction loops
            self.sb.enter_body(ForStmt(axis, extent))

            # compare and update index
            expr = self.visit(value)
            with self.sb.if_then(sc.reduce_operation.arg_combine(lhs_value=expr, rhs_value=acc_value)):
                self.sb += AssignStmt(acc_value, expr)
                self.sb += AssignStmt(acc_index, axis)

            # exit loop
            self.sb.exit_body()

            # if e is in the input buffer, we should write it back
            if e in self.input_map:
                input_var = self.input_map[e]
                self.sb += AssignStmt(input_var, acc_index)

            return acc_index
        else:
            raise NotImplementedError('Compute pattern {}'.format(type(sc).__name__))


def expand_loop(expr: Expr, input_map: Mapping[Union[ScalarNode, TensorNode], Var]):
    expander = LoopExpander(input_map)
    stmt, value = expander.expand(expr)
    return stmt, value


def params_from_task(task: Task) -> List[Var]:
    return [Var(param.name, param.ttype) for param in task.inputs + task.outputs]
