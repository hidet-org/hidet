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
from typing import Dict, List, Optional
import os

import hidet.option
from hidet.ir.compute import TensorNode, GridCompute, TensorInput
from hidet.ir.expr import Expr, Var, TensorElement, tensor_var, tensor_element
from hidet.ir.stmt import BufferStoreStmt
from hidet.ir.func import Function, IRModule
from hidet.ir.task import Task, InverseMap
from hidet.ir.functors import IRRewriter
from hidet.ir.tools import rewrite, collect
from hidet.transforms import FunctionPass
from hidet.graph.ir import FlowGraph, Operator, Tensor
from hidet.utils import strict_zip
from .fused_operator import FusedTask


class PrologueEpilogueRewriter(IRRewriter):
    def __init__(self, fused_task: FusedTask):
        super().__init__()
        self.fused_task: FusedTask = fused_task
        self.fused_graph: FlowGraph = fused_task.fused_graph
        self.anchor_operator: Operator = fused_task.fused_graph.nodes[fused_task.anchor]
        self.anchor_task: Task = self.anchor_operator.task
        self.anchor_inputs: List[Var] = []
        self.anchor_outputs: List[Var] = []

        # declare inputs and outputs of the fused function
        self.graph_params: List[Var] = []
        self.tensor2var: Dict[Tensor, Var] = {}
        for tn, tensor in zip(fused_task.tensor_params, self.fused_graph.inputs + self.fused_graph.outputs):
            var = tensor_var(tn.name, shape=tensor.shape, dtype=tensor.dtype)
            self.graph_params.append(var)
            self.tensor2var[tensor] = var

        # computation relation
        self.graph_input_to_var: Dict[TensorNode, Var] = {}
        self.graph_output_to_var: Dict[TensorNode, Var] = {}
        self.consume: Dict[TensorInput, TensorNode] = {}
        self.reverse_consume: Dict[TensorNode, List[TensorInput]] = {}
        self.input2task: Dict[TensorInput, Task] = {}
        for node in self.fused_graph.nodes:
            for tensor, tensor_node in zip(node.inputs, node.task.inputs):
                if tensor.op is None:
                    self.graph_input_to_var[tensor_node] = self.tensor2var[tensor]
                else:
                    producer: Operator = tensor.op
                    self.consume[tensor_node] = producer.task.outputs[tensor.trace[1]]
                self.input2task[tensor_node] = node.task
        for a, b in self.consume.items():
            if b not in self.reverse_consume:
                self.reverse_consume[b] = []
            self.reverse_consume[b].append(a)
        for output_tensor in self.fused_graph.outputs:
            node, op_output_idx = output_tensor.trace
            output_compute: TensorNode = node.task.outputs[op_output_idx]
            self.graph_output_to_var[output_compute] = self.tensor2var[output_tensor]

    def visit_Function(self, func: Function):
        if func.kind not in ['cuda_kernel', 'host_kernel']:
            return func
        else:
            # extract tensor inputs and outputs of the anchor function
            param_dict: Dict[TensorNode, Var] = {
                task_param: func_param
                for task_param, func_param in zip(self.anchor_task.params, func.params)
                if isinstance(task_param, TensorNode)
            }
            self.anchor_inputs: List[Var] = [param_dict[task_input] for task_input in self.anchor_task.inputs]
            self.anchor_outputs: List[Var] = [param_dict[task_output] for task_output in self.anchor_task.outputs]

            return Function(
                name=func.name,
                params=self.graph_params,
                body=self.visit(func.body),
                ret_type=func.ret_type,
                kind=func.kind,
                extern_vars=func.extern_vars,
                attrs=func.attrs,
            )

    def visit_Var(self, e: Var):
        if e in self.anchor_inputs:
            input_index = self.anchor_inputs.index(e)

            if self.anchor_operator.inputs[input_index].op is None:
                return self.tensor2var[self.anchor_operator.inputs[input_index]]
            else:
                # we encounter a usage of an input tensor of the task other than TensorElement and BufferStoreStmt
                raise ValueError(
                    'Did you used a tensor in expression other than pure tensor indexing (e.g., tensor[...])'
                    ' while marking the task as allowing prologue?'
                )
        elif e in self.anchor_outputs:
            output_index = self.anchor_outputs.index(e)
            if self.anchor_operator.outputs[output_index] in self.fused_graph.outputs:
                return self.tensor2var[self.anchor_operator.outputs[output_index]]
            else:
                # we encounter a usage of an output tensor of the task other than TensorElement and BufferStoreStmt
                raise ValueError(
                    'Did you used a tensor in expression other than tensor storing (e.g., tensor[...] = ...)'
                    ' while marking the task as allowing epilogue?'
                )
        else:
            return e

    def visit_TensorElement(self, e: TensorElement):
        if e.base in self.anchor_inputs:
            # access an input tensor in the anchor operator, replace it with the task input (i.e., InputTensor)
            input_index = self.anchor_inputs.index(e.base)
            return self.visit(tensor_element(self.anchor_task.inputs[input_index], e.indices))
        elif isinstance(e.base, TensorNode):
            # apply prologue
            buf: TensorNode = e.base
            indices = tuple(self.visit(v) for v in e.indices)
            if isinstance(buf, TensorInput):
                if buf in self.graph_input_to_var:
                    # buf is an input tensor of the fused graph
                    return tensor_element(self.graph_input_to_var[buf], indices)
                elif buf in self.consume:
                    # buf is an input tensor of an inner task of the fused graph,
                    # but not an input tensor of fused graph.
                    buf = self.consume[buf]
                    return self.visit(buf[indices])
                else:
                    raise ValueError('Input tensor {} has not been bound.'.format(buf))
            elif isinstance(buf, GridCompute):
                remap = {a: b for a, b in strict_zip(buf.axes, indices)}
                return self.visit(rewrite(buf.value, remap))
            else:
                raise ValueError('Prologue can only use GridCompute primitive.')
        else:
            return IRRewriter.visit_TensorElement(self, e)

    def visit_BufferStoreStmt(self, stmt: BufferStoreStmt):
        if stmt.buf in self.anchor_outputs:
            # store a value to an output tensor with epilogue, replace it with the task output (i.e., TensorNode)
            output_index = self.anchor_outputs.index(stmt.buf)
            return self.visit(
                BufferStoreStmt(self.anchor_task.outputs[output_index], stmt.indices, stmt.value, stmt.protected)
            )
        elif isinstance(stmt.buf, TensorNode):
            # apply epilogue
            buf: TensorNode = stmt.buf
            indices = [self.visit(v) for v in stmt.indices]
            if buf in self.graph_output_to_var:
                # buf is an output tensor of the task graph
                return BufferStoreStmt(self.graph_output_to_var[buf], indices, self.visit(stmt.value), stmt.protected)
            elif buf in self.reverse_consume:
                # buf is an output tensor of an inner task of the task graph,
                # but not an output tensor of task graph.
                consumed_by: List[TensorInput] = self.reverse_consume[buf]
                if len(consumed_by) != 1:
                    raise ValueError(
                        'Expect tensor {} to be consumed exactly once, got {}.'.format(buf, len(consumed_by))
                    )
                consumer_input: TensorInput = consumed_by[0]
                consumer_task: Task = self.input2task[consumer_input]
                inverse_map: InverseMap = consumer_task.inverse_map[consumer_input]
                assert len(consumer_task.outputs) == 1, 'Expect consumer task to have exactly one output.'
                consumer_output: TensorNode = consumer_task.outputs[0]
                assert isinstance(consumer_output, GridCompute), 'Only GridCompute is supported in epilogue.'

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
                gc: GridCompute = consumer_output
                remap: Dict[Var, Expr] = {a: b for a, b in strict_zip(gc.axes, out_indices)}
                value: Expr = rewrite(gc.value, remap)
                # replace out[i + 3, i + j] with value (in the example above)
                tensor_elements: List[TensorElement] = collect(value, TensorElement, stop_when_found=False)
                tensor_elements = [te for te in tensor_elements if te.base is consumer_input]
                assert len(tensor_elements) == 1, (
                    'Epilogue can only index one time of the input tensor ' 'with inverse map'
                )
                te: TensorElement = tensor_elements[0]
                # in the context of above example, we replace 'out[i + 3, i + j]' by 'value'
                self.memo[te] = self.visit(stmt.value)

                # step 3
                return self.visit(BufferStoreStmt(consumer_output, out_indices, value, stmt.protected))
            else:
                raise ValueError('Output tensor {} has not been bound.'.format(buf))
        else:
            return IRRewriter.visit_BufferStoreStmt(self, stmt)


class FusePrologueEpiloguePass(FunctionPass):
    def __init__(self, fused_task: FusedTask):
        super().__init__()
        self.rewriter = PrologueEpilogueRewriter(fused_task)

    def process_func(self, func: Function) -> Function:
        return self.rewriter.visit(func)


def fuse_prologue_epilogue_pass(fused_task: FusedTask):
    return FusePrologueEpiloguePass(fused_task)


def apply_prologue_epilogue(ir_module: IRModule, fused_task: FusedTask, working_dir: str) -> IRModule:
    from hidet.transforms import inline_function_pass, declare_to_let_pass, inline_let_stmt_pass
    from hidet.transforms import flatten_tensor_slice_pass, lower_with, PassContext, SaveIRInstrument, ProfileInstrument

    anchor_function: Optional[Function] = None
    for func in ir_module.functions.values():
        if func.kind in ['cuda_kernel', 'host_kernel']:
            if anchor_function is not None:
                raise RuntimeError('More than one function found.')
            anchor_function = func
    if anchor_function is None:
        raise RuntimeError('No kernel function found.')

    transforms = [
        flatten_tensor_slice_pass(),
        inline_function_pass(),
        declare_to_let_pass(),
        inline_let_stmt_pass(inline_all=False),
        fuse_prologue_epilogue_pass(fused_task),
    ]
    instruments = []
    if hidet.option.get_save_lower_ir():
        instruments.append(SaveIRInstrument(out_dir=os.path.join(working_dir, './fuse_ir')))
        instruments.append(ProfileInstrument(log_file=os.path.join(working_dir, './fuse_ir/profile.txt')))

    with PassContext(instruments=instruments):
        ir_module = lower_with(ir_module, transforms)

    return ir_module
