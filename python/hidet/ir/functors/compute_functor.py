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
# pylint: disable=bad-staticmethod-argument, too-many-boolean-expressions
from typing import Any, Union, List, Dict, Tuple

from hidet.ir.task import Task
from hidet.ir.compute import TensorInput, ScalarInput, ReduceCompute, ArgReduceCompute, GridCompute
from hidet.ir.node import Node
from hidet.utils import same_list
from .base_functor import BaseFunctor, BaseVisitor, BaseRewriter


class ComputeFunctor(BaseFunctor):
    def visit_dispatch(self, node: Union[Node, Tuple, List, Dict[str, Any], str]):
        if isinstance(node, Task):
            return self.visit_Task(node)
        elif isinstance(node, ScalarInput):
            return self.visit_ScalarInput(node)
        elif isinstance(node, TensorInput):
            return self.visit_TensorInput(node)
        elif isinstance(node, GridCompute):
            return self.visit_GridCompute(node)
        elif isinstance(node, ReduceCompute):
            return self.visit_ReduceCompute(node)
        elif isinstance(node, ArgReduceCompute):
            return self.visit_ArgReduceCompute(node)
        else:
            return NotImplemented

    def visit_Task(self, task: Task):
        raise NotImplementedError()

    def visit_ScalarInput(self, node: ScalarInput):
        raise NotImplementedError()

    def visit_TensorInput(self, node: TensorInput):
        raise NotImplementedError()

    def visit_GridCompute(self, node: GridCompute):
        raise NotImplementedError()

    def visit_ReduceCompute(self, node: ReduceCompute):
        raise NotImplementedError()

    def visit_ArgReduceCompute(self, node: ArgReduceCompute):
        raise NotImplementedError()


class ComputeVisitor(BaseVisitor, ComputeFunctor):
    def visit_Task(self, task: Task):
        self.visit(task.inputs)
        self.visit(task.outputs)

    def visit_ScalarInput(self, node: ScalarInput):
        self.visit(node.dtype)

    def visit_TensorInput(self, node: TensorInput):
        self.visit(node.ttype)

    def visit_GridCompute(self, node: GridCompute):
        self.visit(node.shape)
        self.visit(node.axes)
        self.visit(node.value)
        self.visit(node.layout)
        self.visit(node.input_scalars)
        self.visit(node.input_tensors)

    def visit_ReduceCompute(self, node: ReduceCompute):
        self.visit(node.shape)
        self.visit(node.axes)
        self.visit(node.value)
        self.visit(node.accumulate_dtype)
        self.visit(node.input_scalars)
        self.visit(node.input_tensors)

    def visit_ArgReduceCompute(self, node: ArgReduceCompute):
        self.visit(node.extent)
        self.visit(node.axis)
        self.visit(node.value)
        self.visit(node.index_dtype)
        self.visit(node.input_scalars)
        self.visit(node.input_tensors)


class ComputeRewriter(BaseRewriter, ComputeFunctor):
    def visit_Task(self, task: Task):
        return task

    def visit_ScalarInput(self, node: ScalarInput):
        dtype = self.visit(node.dtype)
        if dtype is node.dtype:
            return node
        else:
            return ScalarInput(node.name, dtype)

    def visit_TensorInput(self, node: TensorInput):
        ttype = self.visit(node.ttype)
        if ttype is node.ttype:
            return node
        else:
            return TensorInput(node.name, ttype)

    def visit_GridCompute(self, node: GridCompute):
        shape = self.visit(node.shape)
        axes = self.visit(node.axes)
        value = self.visit(node.value)
        layout = self.visit(node.layout)
        input_scalars = self.visit(node.input_scalars)
        input_tensors = self.visit(node.input_tensors)
        if (
            value is node.value
            and layout is node.layout
            and same_list(shape, node.shape)
            and same_list(axes, node.axes)
            and same_list(input_tensors, node.input_tensors)
            and same_list(input_scalars, node.input_scalars)
        ):
            return node
        else:
            return GridCompute(node.name, input_tensors, input_scalars, shape, axes, value, layout)

    def visit_ReduceCompute(self, node: ReduceCompute):
        shape = self.visit(node.shape)
        axes = self.visit(node.axes)
        value = self.visit(node.value)
        accumulate_dtype = self.visit(node.accumulate_dtype)
        input_scalars = self.visit(node.input_scalars)
        input_tensors = self.visit(node.input_tensors)
        if (
            value is node.value
            and accumulate_dtype is node.accumulate_dtype
            and same_list(shape, node.shape)
            and same_list(axes, node.axes)
            and same_list(input_tensors, node.input_tensors)
            and same_list(input_scalars, node.input_scalars)
        ):
            return node
        else:
            return ReduceCompute(
                node.name, input_tensors, input_scalars, shape, axes, value, node.reduce_operation, accumulate_dtype
            )

    def visit_ArgReduceCompute(self, node: ArgReduceCompute):
        extent = self.visit(node.extent)
        axis = self.visit(node.axis)
        value = self.visit(node.value)
        index_dtype = self.visit(node.index_dtype)
        input_scalars = self.visit(node.input_scalars)
        input_tensors = self.visit(node.input_tensors)
        if (
            value is node.value
            and index_dtype is node.index_dtype
            and extent is node.extent
            and axis is node.axis
            and same_list(input_tensors, node.input_tensors)
            and same_list(input_scalars, node.input_scalars)
        ):
            return node
        else:
            return ArgReduceCompute(
                node.name, input_tensors, input_scalars, extent, axis, value, node.reduce_operation, index_dtype
            )
