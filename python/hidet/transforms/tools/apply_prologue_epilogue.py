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
from typing import Sequence, Dict, List
from hidet.ir.compute import TensorNode, GridCompute, TensorCompute
from hidet.ir.expr import Expr, Var, TensorElement
from hidet.ir.stmt import BufferStoreStmt
from hidet.ir.func import Function, IRModule
from hidet.ir.task import Task, TaskGraph, InverseMap
from hidet.ir.functors import FuncStmtExprRewriter, ExprRewriter, rewrite, collect
from hidet.utils import strict_zip


class PrologueIndexer(ExprRewriter):
    def __init__(self):
        super().__init__()
        self.bind: Dict[TensorNode, Var] = {}

    def init_bind(self, input_tensors: Sequence[TensorNode], param_tensors: Sequence[Var]):
        self.bind = {a: b for a, b in strict_zip(input_tensors, param_tensors)}

    def tensor_index(self, buf: TensorNode, indices: Sequence[Expr]) -> Expr:
        tc = buf.tensor_compute
        if tc is None:
            if buf not in self.bind:
                raise ValueError('Input tensor {} has not been bound.'.format(buf))
            return TensorElement(self.bind[buf], indices)
        elif isinstance(tc, GridCompute):
            gc: GridCompute = tc
            remap = {a: b for a, b in strict_zip(gc.axes, indices)}
            return self.visit(rewrite(gc.value, remap))
        else:
            raise ValueError('Prologue can only use GridCompute primitive.')

    def visit_TensorElement(self, e: TensorElement):
        if isinstance(e.base, TensorNode):
            return self.tensor_index(e.base, e.indices)
        else:
            return ExprRewriter.visit_TensorElement(self, e)


class PrologueEpilogueRewriter(FuncStmtExprRewriter):
    def __init__(self, task: Task):
        super().__init__()
        self.task: Task = task
        self.task_graph: TaskGraph = task.task_graph
        self.reverse_consume: Dict[TensorNode, List[TensorNode]] = {}
        for a, b in self.task_graph.consume.items():
            if b not in self.reverse_consume:
                self.reverse_consume[b] = []
            self.reverse_consume[b].append(a)
        self.input2task: Dict[TensorNode, Task] = {}
        for internal_task in self.task_graph.nodes:
            for input_tensor in internal_task.inputs:
                self.input2task[input_tensor] = internal_task

        self.binding: Dict[TensorNode, Var] = {}
        self.new_params: List[Var] = []
        self.anchor_inputs: List[Var] = []
        self.anchor_outputs: List[Var] = []
        self.anchor_input_new_index: Dict[int, int] = {}
        self.anchor_output_new_index: Dict[int, int] = {}

    def visit_Function(self, func: Function):
        anchor_num_inputs = len(self.task.inputs)
        anchor_num_outputs = len(self.task.outputs)
        assert len(func.params) == anchor_num_inputs + anchor_num_outputs
        self.anchor_inputs: List[Var] = func.params[:anchor_num_inputs]
        self.anchor_outputs: List[Var] = func.params[anchor_num_inputs:]

        for input_index in range(anchor_num_inputs):
            tn = self.task.inputs[input_index]
            while tn in self.task_graph.consume:
                tn = self.task_graph.consume[tn]
            if tn in self.task_graph.input_tensors:
                self.anchor_input_new_index[input_index] = self.task_graph.input_tensors.index(tn)

        for output_index in range(anchor_num_outputs):
            tn = self.task.outputs[output_index]
            if tn in self.task_graph.output_tensors:
                self.anchor_output_new_index[output_index] = self.task_graph.output_tensors.index(tn)

        # create parameters for fused function, and bind task graph parameters to function parameters
        # todo: do not create new parameters for the inputs/outputs that have not been fused
        self.new_params: List[Var] = []
        for tensor_node in self.task_graph.input_tensors + self.task_graph.output_tensors:
            self.new_params.append(Var(tensor_node.name, tensor_node.ttype))
            self.binding[tensor_node] = self.new_params[-1]

        return Function(
            name=func.name,
            params=self.new_params,
            body=self.visit(func.body),
            ret_type=func.ret_type,
            kind=func.kind,
            extern_vars=func.extern_vars,
            attrs=func.attrs,
        )

    def visit_Var(self, e: Var):
        if e in self.anchor_inputs:
            input_index = self.anchor_inputs.index(e)
            if input_index in self.anchor_input_new_index:
                return self.new_params[self.anchor_input_new_index[input_index]]
            else:
                # we encounter a usage of an input tensor of the task other than TensorElement and BufferStoreStmt
                raise ValueError(
                    'Did you used a tensor in expression other than tensor[...] and tensor[...] = ...'
                    ' while marking the task as allowing prologue?'
                )
        elif e in self.anchor_outputs:
            output_index = self.anchor_outputs.index(e)
            if output_index in self.anchor_output_new_index:
                return self.new_params[len(self.task_graph.input_tensors) + self.anchor_output_new_index[output_index]]
            else:
                # we encounter a usage of an output tensor of the task other than TensorElement and BufferStoreStmt
                raise ValueError(
                    'Did you used a tensor in expression other than tensor[...] and tensor[...] = ...'
                    ' while marking the task as allowing epilogue?'
                )
        else:
            return e

    def visit_TensorElement(self, e: TensorElement):
        if isinstance(e.base, TensorNode):
            # apply prologue
            buf: TensorNode = e.base
            indices = [self.visit(v) for v in e.indices]
            if buf.tensor_compute is None:
                if buf in self.binding:
                    # buf is an input tensor of the task graph
                    return TensorElement(self.binding[buf], indices)
                elif buf in self.task_graph.consume:
                    # buf is an input tensor of an inner task of the task graph,
                    # but not an input tensor of task graph.
                    buf = self.task_graph.consume[buf]
                    return self.visit(buf[indices])
                else:
                    raise ValueError('Input tensor {} has not been bound.'.format(buf))
            elif isinstance(buf.tensor_compute, GridCompute):
                gc: GridCompute = buf.tensor_compute
                remap = {a: b for a, b in strict_zip(gc.axes, indices)}
                return self.visit(rewrite(gc.value, remap))
            else:
                raise ValueError('Prologue can only use GridCompute primitive.')
        elif e.base in self.anchor_inputs:
            input_index = self.anchor_inputs.index(e.base)
            return self.visit(TensorElement(self.task.inputs[input_index], e.indices))
        else:
            return FuncStmtExprRewriter.visit_TensorElement(self, e)

    def visit_BufferStoreStmt(self, stmt: BufferStoreStmt):
        if isinstance(stmt.buf, TensorNode):
            # apply epilogue
            buf: TensorNode = stmt.buf
            indices = [self.visit(v) for v in stmt.indices]
            if buf in self.task_graph.output_tensors:
                # buf is an output tensor of the task graph
                return BufferStoreStmt(self.binding[buf], indices, self.visit(stmt.value), stmt.protected)
            elif buf in self.reverse_consume:
                # buf is an output tensor of an inner task of the task graph,
                # but not an output tensor of task graph.
                consumed_by: List[TensorNode] = self.reverse_consume[buf]
                if len(consumed_by) != 1:
                    raise ValueError(
                        'Expect tensor {} to be consumed exactly once, got {}.'.format(buf, len(consumed_by))
                    )
                consumer_input: TensorNode = consumed_by[0]
                consumer_task: Task = self.input2task[consumer_input]
                inverse_map: InverseMap = consumer_task.inverse_map[consumer_input]
                assert len(consumer_task.outputs) == 1, 'Expect consumer task to have exactly one output.'
                consumer_output: TensorNode = consumer_task.outputs[0]

                # Example of what we are doing here:
                # original indices:
                # epilogue_out[i, j] = expr(i, j, out[i + 3, i + j])
                # inverse_map: (p, q) -> (p - 3, q - p - 3)
                #
                # original statement:
                # out[e1, e2] = value (e1, e2, and value are all Expr)
                #
                # expected statement:
                # epilogue_out[e1 - 3, e2 - e1 - 3] = expr(e1 - 3, e2 - e1 - 3, value)
                #
                # steps to get the expected statement:
                # 1. get the output index expressions using inverse_map
                #    e.g., e1 - 3, e2 - e1 - 3
                # 2. get the value expression to be stored
                #    e.g. expr(e1 - 3, e2 - e1 - 3, value)
                # 3. create the expected statement. If it still has epilogue, repeat above steps repeatedly.

                # step 1
                remap: Dict[Var, Expr] = {a: b for a, b in strict_zip(inverse_map.axes, indices)}
                out_indices: List[Expr] = [rewrite(e, remap) for e in inverse_map.indices]

                # step 2
                # replace index
                tc: TensorCompute = consumer_output.tensor_compute
                assert isinstance(tc, GridCompute), 'Only GridCompute is supported in epilogue, got {}.'.format(tc)
                remap: Dict[Var, Expr] = {a: b for a, b in strict_zip(tc.axes, out_indices)}
                value: Expr = rewrite(tc.value, remap)
                # replace out[i + 3, i + j] with value (in the example above)
                tensor_elements: List[TensorElement] = collect(value, TensorElement, stop_when_found=False)
                tensor_elements = [te for te in tensor_elements if te.base is consumer_input]
                assert len(tensor_elements) == 1, (
                    'Epilgoue can only index one time of the input tensor ' 'with inverse map'
                )
                tensor_element: TensorElement = tensor_elements[0]
                # in the context of above example, we replace 'out[i + 3, i + j]' by 'value'
                self.memo[tensor_element] = self.visit(stmt.value)

                # step 3
                return self.visit(BufferStoreStmt(consumer_output, out_indices, value, stmt.protected))
            else:
                raise ValueError('Output tensor {} has not been bound.'.format(buf))
        elif stmt.buf in self.anchor_outputs:
            output_index = self.anchor_outputs.index(stmt.buf)
            return self.visit(
                BufferStoreStmt(self.task.outputs[output_index], stmt.indices, stmt.value, stmt.protected)
            )
        else:
            return FuncStmtExprRewriter.visit_BufferStoreStmt(self, stmt)


def apply_prologue_epilogue(ir_module: IRModule, func: Function, task: Task) -> Function:
    from hidet.transforms.flatten_tensor_slice import FlattenTensorSliceRewriter

    rewriter1 = FlattenTensorSliceRewriter()
    rewriter2 = PrologueEpilogueRewriter(task)
    fused_func = rewriter2(rewriter1(func))
    ir_module.update_function(fused_func)
    return fused_func
