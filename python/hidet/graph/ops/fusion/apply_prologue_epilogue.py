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
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import os

import hidet.option
from hidet.ir.compute import TensorNode, GridCompute, TensorInput
from hidet.ir.expr import Expr, Var, TensorElement, Call, tensor_element, var, tensor_pointer_var
from hidet.ir.stmt import BufferStoreStmt, LaunchKernelStmt
from hidet.ir.func import Function
from hidet.ir.module import IRModule
from hidet.ir.task import Task, InverseMap
from hidet.ir.functors import IRRewriter, IRVisitor
from hidet.ir.tools import rewrite, collect
from hidet.ir.utils.call_graph import CallGraph
from hidet.transforms import Pass
from hidet.graph import FlowGraph, Operator, Tensor
from hidet.utils import strict_zip
from .fused_operator import FusedTask


class Prologue:
    """
    An input tensor of an operator can have a prologue, which defines how each element of the input tensor is computed.

    For example, consider the following operator:
    Subgraph with two operators a and b:
        a[i, j] = i * j + i - j (0 <= i < 10, 0 <= j < 10)
        b[i, j] = a[j, i] + a[i, j] (0 <= i < 10, 0 <= j < 10)

        If we take the operator b as the anchor operator, the prologue of b is:
            inputs: [a]
            axes: [i, j]
            expr: a[j, i] + a[i, j]
    """

    def __init__(self, inputs: List[TensorInput], axes: List[Var], expr: Expr):
        self.inputs: List[TensorInput] = inputs
        self.axes: List[Var] = axes
        self.expr: Expr = expr


class Epilogue:
    """
    An output tensor of an operator can have an epilogue, which defines how each element of the output tensor can be
    further computed to produce the result of an element of the output tensor of a sub-graph.

    For example, consider the following operator:
    Subgraph with two operators a and b:
        a[i, j] = i * j + i - j (0 <= i < 10, 0 <= j < 10)
        b[i, j] = a[j, i] + i - j (0 <= i < 10, 0 <= j < 10)

        Then, if we take the operator a as the anchor operator, the epilogue of a is:
            inputs: []
            axes: [i, j]
            value: v
            out_tensor: b
            out_indices: [j, i]
            out_expr: v + j - i  (pay attention to the sign of i and j)
    """

    def __init__(
        self,
        inputs: List[TensorInput],
        axes: List[Var],
        value: Var,
        out_tensor: Tensor,
        out_indices: List[Expr],
        out_expr: Expr,
    ):
        self.inputs: List[TensorInput] = inputs
        self.axes: List[Var] = axes
        self.value: Var = value
        self.out_tensor: Tensor = out_tensor
        self.out_indices: List[Expr] = out_indices
        self.out_expr: Expr = out_expr


class PrologueEpilogueExtractor(IRRewriter):
    def __init__(self, fused_task: FusedTask):
        super().__init__()
        self.graph: FlowGraph = fused_task.fused_graph
        self.anchor_operator: Operator = fused_task.fused_graph.nodes[fused_task.anchor]
        self.anchor_task: Task = self.anchor_operator.task

        self.tensor_map: Dict[TensorNode, Tensor] = {}
        for node in self.graph.nodes:
            for task_tensor, tensor in zip(node.task.params, node.inputs + node.outputs):
                self.tensor_map[task_tensor] = tensor

        self.consume: Dict[TensorInput, TensorNode] = {}
        self.consume_by: Dict[TensorNode, List[TensorInput]] = {}
        for node in self.graph.nodes:
            for task_input, tensor in zip(node.task.inputs, node.inputs):
                if tensor.trace is None:
                    continue
                producer: Operator = tensor.op
                producer_output = producer.task.outputs[tensor.trace[1]]
                self.consume[task_input] = producer_output
                if producer_output not in self.consume_by:
                    self.consume_by[producer_output] = []
                self.consume_by[producer_output].append(task_input)

        self.input2task: Dict[TensorInput, Task] = {}
        for node in self.graph.nodes:
            for task_input in node.task.inputs:
                self.input2task[task_input] = node.task

    def extract(self) -> Tuple[Dict[Tensor, Prologue], Dict[Tensor, Epilogue], Dict[TensorNode, Tensor]]:
        """
        Extract prologues and epilogues from the fused graph.
        """
        # extract prologues
        prologues: Dict[Tensor, Prologue] = {}
        for task_input, tensor in zip(self.anchor_task.inputs, self.anchor_operator.inputs):
            if self.tensor_map[task_input] in self.graph.inputs:
                # this input does not have a prologue, skip
                continue
            axes = [var('i') for _ in range(len(task_input.shape))]
            te = tensor_element(task_input, axes)
            expr: Expr = self.visit(te)
            used_tensor_inputs = collect(expr, TensorInput)
            prologues[tensor] = Prologue(inputs=used_tensor_inputs, axes=axes, expr=expr)

        # extract epilogues
        epilogues: Dict[Tensor, Epilogue] = {}
        for task_output, tensor in zip(self.anchor_task.outputs, self.anchor_operator.outputs):
            axes = [var('i') for _ in range(len(task_output.shape))]
            value = var('value', task_output.type.dtype)
            bss = BufferStoreStmt(buf=task_output, indices=axes, value=value)
            updated_bss = self.visit(bss)
            assert isinstance(updated_bss, BufferStoreStmt)
            used_tensor_inputs = collect(updated_bss.value, TensorInput)
            epilogues[tensor] = Epilogue(
                inputs=used_tensor_inputs,
                axes=axes,
                value=value,
                out_tensor=self.tensor_map[updated_bss.buf],
                out_indices=updated_bss.indices,
                out_expr=updated_bss.value,
            )

        return prologues, epilogues, self.tensor_map

    def visit_TensorElement(self, e: TensorElement):
        indices: Tuple[Expr, ...] = self.visit(e.indices)
        if isinstance(e.base, TensorInput):
            if e.base not in self.tensor_map:
                raise ValueError(f'Cannot find tensor for input {e.base}')
            tensor: Tensor = self.tensor_map[e.base]
            if tensor in self.graph.inputs:
                # the graph input, directly use the tensor
                return tensor_element(e.base, indices)
            else:
                # intermediate tensor
                op, output_idx = tensor.trace
                return self.visit(tensor_element(op.task.outputs[output_idx], indices))
        elif isinstance(e.base, GridCompute):
            remap = {a: b for a, b in zip(e.base.axes, indices)}
            return self.visit(rewrite(e.base.value, remap))
        else:
            raise NotImplementedError(f'Unsupported tensor {e}')

    def visit_BufferStoreStmt(self, stmt: BufferStoreStmt):
        indices: Tuple[Expr, ...] = self.visit(stmt.indices)
        stmt_value: Expr = self.visit(stmt.value)
        if isinstance(stmt.buf, GridCompute):
            tensor = self.tensor_map[stmt.buf]
            if tensor in self.graph.outputs:
                # the graph output
                return BufferStoreStmt(buf=stmt.buf, indices=indices, value=stmt_value)
            else:
                # get the consumer TensorInput, Task, and its GridCompute output
                assert len(self.consume_by[stmt.buf]) == 1, 'Expect only one consumer for {}.'.format(stmt.buf)
                consumer_input: TensorInput = self.consume_by[stmt.buf][0]

                consumer_task: Task = self.input2task[consumer_input]
                assert len(consumer_task.outputs) == 1, 'Expect consumer task to have exactly one output.'

                consumer_output: TensorNode = consumer_task.outputs[0]
                assert isinstance(consumer_output, GridCompute), 'Only GridCompute is supported in epilogue.'
                gc: GridCompute = consumer_output

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
                inverse_map: InverseMap = consumer_task.inverse_map[consumer_input]
                remap: Dict[Var, Expr] = {a: b for a, b in strict_zip(inverse_map.axes, indices)}
                out_indices: List[Expr] = [rewrite(e, remap) for e in inverse_map.indices]

                # step 2
                # replace index
                remap: Dict[Var, Expr] = {a: b for a, b in strict_zip(gc.axes, out_indices)}
                gc_value: Expr = rewrite(gc.value, remap)
                # replace out[i + 3, i + j] with value (in the example above)
                tensor_elements: List[TensorElement] = collect(gc_value, TensorElement, stop_when_found=False)
                tensor_elements = [te for te in tensor_elements if te.base is consumer_input]
                assert (
                    len(tensor_elements) == 1
                ), 'Epilogue can only index one time of the input tensor with inverse map'
                te: TensorElement = tensor_elements[0]
                # in the context of above example, we replace 'out[i + 3, i + j]' by 'value'
                self.memo[te] = stmt_value

                # step 3
                return self.visit(BufferStoreStmt(consumer_output, out_indices, gc_value))
        else:
            raise NotImplementedError(f'Unsupported buffer {stmt.buf}')


class PrologueEpilogueMarker(IRVisitor):
    def __init__(self, fused_task: FusedTask, prologues: Dict[Tensor, Prologue], epilogues: Dict[Tensor, Epilogue]):
        super().__init__()
        self.anchor: Operator = fused_task.fused_graph.nodes[fused_task.anchor]
        self.prologues: Dict[Tensor, Prologue] = prologues
        self.epilogues: Dict[Tensor, Epilogue] = epilogues
        self.param_match: Dict[str, Dict[Var, Tensor]] = defaultdict(dict)
        self.ir_module: Optional[IRModule] = None
        self.caller_name: Optional[str] = None

    def mark(self, module: IRModule) -> Dict[str, Dict[Var, Tensor]]:
        """
        Get the mapping from the parameters of each function to the corresponding tensors in FlowGraph, if that
        parameter represents a tensor.

        Examples
        --------

        .. code-block:: python

            def launch(a, b, c):
                declare d
                f(a, b, d)
                g(d, c)

        would return
        {'launch': {a: tensor_a, b: tensor_b, c: tensor_c},
          'f': {a: tensor_a, b: tensor_b},
          'g': {c: tensor_c}}

        """
        self.visit_IRModule(module)
        return self.param_match

    def visit_IRModule(self, module: IRModule):
        self.ir_module = module

        if 'launch' not in module.functions:
            raise ValueError('Cannot find launch function in the module.')
        call_graph = CallGraph(module, allow_missing=True)

        # initialize param_match for launch function
        launch_func: Function = module.functions['launch']
        for param_var, tensor in zip(launch_func.params, self.anchor.inputs + self.anchor.outputs):
            self.param_match[launch_func.name][param_var] = tensor

        # fill in param_match for callee functions
        for node in call_graph.order:
            func: Function = node.func
            self.caller_name = func.name
            self.visit(func)

    def process_call(self, callee_name, args):
        if callee_name not in self.ir_module.functions:
            return
        callee: Function = self.ir_module.functions[callee_name]
        for param_var, arg in zip(callee.params, args):
            if arg in self.param_match[self.caller_name]:
                tensor = self.param_match[self.caller_name][arg]
                if param_var not in self.param_match[callee_name]:
                    self.param_match[callee_name][param_var] = tensor
                else:
                    if self.param_match[callee_name][param_var] is not tensor:
                        msg = f'Inconsistent tensor node {tensor} for param {param_var} in function {callee_name}'
                        raise ValueError(msg)

    def visit_Call(self, e: Call):
        self.process_call(e.func_var.name, e.args)

    def visit_LaunchKernelStmt(self, stmt: LaunchKernelStmt):
        self.process_call(stmt.func_var.name, stmt.args)


class FuncParamRecord:
    def __init__(self, param_to_tensor, updated_params, tensor_to_updated_param):
        # the original param to the corresponding input/output tensor of the anchor operator
        self.param_to_tensor: Dict[Var, Tensor] = param_to_tensor

        # the updated parameters and mapping from the input/output tensor of the fused operator to updated params
        self.updated_params: List[Var] = updated_params
        self.tensor_to_updated_param: Dict[Tensor, Var] = tensor_to_updated_param


class PrologueEpilogueRewriter(IRRewriter):
    def __init__(
        self,
        fused_task: FusedTask,
        prologues: Dict[Tensor, Prologue],
        epilogues: Dict[Tensor, Epilogue],
        tensor_map: Dict[TensorNode, Tensor],
        marks: Dict[str, Dict[Var, Tensor]],
    ):
        super().__init__()
        self.fused_graph: FlowGraph = fused_task.fused_graph
        self.anchor: Operator = fused_task.fused_graph.nodes[fused_task.anchor]
        self.prologues: Dict[Tensor, Prologue] = prologues
        self.epilogues: Dict[Tensor, Epilogue] = epilogues
        self.marks: Dict[str, Dict[Var, Tensor]] = marks
        self.tensor_map: Dict[TensorNode, Tensor] = tensor_map

        self.func_records: Dict[str, FuncParamRecord] = {}
        self.current_record: Optional[FuncParamRecord] = None
        self.ir_module: Optional[IRModule] = None

    def visit_IRModule(self, module: IRModule):
        call_graph = CallGraph(module, allow_missing=True)
        self.ir_module = module

        for func_name, var2tensor in self.marks.items():
            param_to_tensor: Dict[Var, Tensor] = {}
            # the original param to the corresponding input/output tensor of the anchor operator
            for param_var, anchor_tensor in var2tensor.items():
                param_to_tensor[param_var] = anchor_tensor

            # add the parameters that does not correspond to any tensor, or has no prologue/epilogue
            updated_params = []
            tensor_to_updated_param: Dict[Tensor, Var] = {}
            for param in self.ir_module.functions[func_name].params:
                if param in param_to_tensor:
                    tensor = param_to_tensor[param]
                    if tensor not in self.prologues and tensor not in self.epilogues:
                        updated_params.append(param)
                        tensor_to_updated_param[tensor] = param
                    else:
                        # skip the parameters that have prologue/epilogue, since they will not exist
                        pass
                else:
                    # the parameter does not correspond to any tensor, keep it
                    updated_params.append(param)

            # add the required input/output tensors for prologue/epilogue
            for param_var, tensor in param_to_tensor.items():
                if tensor in self.prologues:
                    prologue: Prologue = self.prologues[tensor]
                    for ti in prologue.inputs:
                        t = self.tensor_map[ti]
                        if t not in tensor_to_updated_param:
                            updated_params.append(tensor_pointer_var(ti.name, t.shape, t.dtype))
                            tensor_to_updated_param[t] = updated_params[-1]
                elif tensor in self.epilogues:
                    epilogue: Epilogue = self.epilogues[tensor]
                    for ti in epilogue.inputs:
                        t = self.tensor_map[ti]
                        if t not in tensor_to_updated_param:
                            updated_params.append(tensor_pointer_var(ti.name, t.shape, t.dtype))
                            tensor_to_updated_param[t] = updated_params[-1]
                    out_tensor = epilogue.out_tensor
                    updated_params.append(tensor_pointer_var(param_var.hint, out_tensor.shape, out_tensor.dtype))
                    tensor_to_updated_param[out_tensor] = updated_params[-1]
                else:
                    # the parameter does not have prologue/epilogue, skip it
                    pass

            if func_name == 'launch':
                # reorder the updated params to match the order of the inputs and outputs of the graph
                assert len(updated_params) == len(self.fused_graph.inputs) + len(self.fused_graph.outputs)
                updated_params = [
                    tensor_to_updated_param[t] for t in self.fused_graph.inputs + self.fused_graph.outputs
                ]

            self.func_records[func_name] = FuncParamRecord(param_to_tensor, updated_params, tensor_to_updated_param)

        for node in call_graph.order:
            func: Function = node.func
            self.visit(func)

        return module

    def visit_Function(self, func: Function):
        if func.name not in self.func_records:
            return func

        record: FuncParamRecord = self.func_records[func.name]

        self.current_record = record

        # update the parameters of the function
        func.params = record.updated_params
        func.body = self.visit(func.body)
        return func

    def visit_Var(self, e: Var):
        if e in self.current_record.param_to_tensor:
            tensor = self.current_record.param_to_tensor[e]
            if tensor in self.prologues:
                # we encounter a usage of an input tensor of the task other than TensorElement and BufferStoreStmt
                raise ValueError(
                    'Did you used a tensor in expression other than pure tensor indexing (e.g., tensor[...])'
                    ' while marking the task as allowing prologue?'
                )
            elif tensor in self.epilogues:
                # we encounter a usage of an output tensor of the task other than TensorElement and BufferStoreStmt
                raise ValueError(
                    'Did you used a tensor in expression other than tensor storing (e.g., tensor[...] = ...)'
                    ' while marking the task as allowing epilogue?'
                )
        return super().visit_Var(e)

    def process_call(self, func_name: str, args: List[Expr]) -> List[Expr]:
        caller_record: FuncParamRecord = self.current_record
        callee_record: FuncParamRecord = self.func_records[func_name]
        new_args = []
        origin_param_to_arg: Dict[Var, Expr] = {a: b for a, b in zip(self.ir_module.functions[func_name].params, args)}
        updated_var_to_tensor: Dict[Var, Tensor] = {v: k for k, v in callee_record.tensor_to_updated_param.items()}
        for updated_param in callee_record.updated_params:
            if updated_param in updated_var_to_tensor:
                new_args.append(caller_record.tensor_to_updated_param[updated_var_to_tensor[updated_param]])
            else:
                assert updated_param in origin_param_to_arg
                new_args.append(self.visit(origin_param_to_arg[updated_param]))
        return new_args

    def visit_Call(self, e: Call):
        if isinstance(e.func_var, Var):
            func_name = e.func_var.name
            if func_name in self.func_records:
                args = self.process_call(func_name, list(e.args))
                return Call(e.func_var, args)
        return super().visit_Call(e)

    def visit_LaunchKernelStmt(self, stmt: LaunchKernelStmt):
        if isinstance(stmt.func_var, Var):
            func_name = stmt.func_var.name
            if func_name in self.func_records:
                args = self.process_call(func_name, list(stmt.args))
                return LaunchKernelStmt(stmt.func_var, args, stmt.grid_dim, stmt.block_dim, stmt.shared_mem_bytes)
        return super().visit_LaunchKernelStmt(stmt)

    def visit_TensorElement(self, e: TensorElement):
        if e.base in self.current_record.param_to_tensor:
            assert isinstance(e.base, Var)
            tensor: Tensor = self.current_record.param_to_tensor[e.base]
            if tensor in self.prologues:
                prologue: Prologue = self.prologues[tensor]
                remap = {ti: self.current_record.tensor_to_updated_param[self.tensor_map[ti]] for ti in prologue.inputs}
                remap.update({a: b for a, b in zip(prologue.axes, self.visit(e.indices))})
                return rewrite(prologue.expr, remap)

        return super().visit_TensorElement(e)

    def visit_BufferStoreStmt(self, stmt: BufferStoreStmt):
        if stmt.buf in self.current_record.param_to_tensor:
            tensor: Tensor = self.current_record.param_to_tensor[stmt.buf]
            if tensor in self.epilogues:
                epilogue: Epilogue = self.epilogues[tensor]
                indices = self.visit(stmt.indices)
                value = self.visit(stmt.value)

                remap = {ti: self.current_record.tensor_to_updated_param[self.tensor_map[ti]] for ti in epilogue.inputs}
                remap.update({a: b for a, b in zip(epilogue.axes, indices)})
                remap.update({epilogue.value: value})
                out_indices = rewrite(epilogue.out_indices, remap)
                out_expr = rewrite(epilogue.out_expr, remap)
                out_buf = self.current_record.tensor_to_updated_param[epilogue.out_tensor]
                return BufferStoreStmt(out_buf, out_indices, out_expr)

        return super().visit_BufferStoreStmt(stmt)


class FusePrologueEpiloguePass(Pass):
    def __init__(self, fused_task: FusedTask):
        super().__init__()
        self.fused_task: FusedTask = fused_task

    def process_module(self, ir_module: IRModule) -> IRModule:
        extractor = PrologueEpilogueExtractor(self.fused_task)
        prologues, epilogues, tensor_map = extractor.extract()

        marker = PrologueEpilogueMarker(self.fused_task, prologues, epilogues)
        marks: Dict[str, Dict[Var, Tensor]] = marker.mark(ir_module)

        rewriter = PrologueEpilogueRewriter(self.fused_task, prologues, epilogues, tensor_map, marks)
        return rewriter.rewrite(ir_module)


def fuse_prologue_epilogue_pass(fused_task: FusedTask):
    return FusePrologueEpiloguePass(fused_task)


def apply_prologue_epilogue(ir_module: IRModule, fused_task: FusedTask, working_dir: str) -> IRModule:
    from hidet.transforms import inline_function_pass, declare_to_let_pass, inline_let_stmt_pass
    from hidet.transforms import flatten_tensor_slice_pass, lower_with, PassContext, SaveIRInstrument, ProfileInstrument
    from hidet.transforms import generate_launch_func_pass

    transforms = [
        generate_launch_func_pass(),
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
