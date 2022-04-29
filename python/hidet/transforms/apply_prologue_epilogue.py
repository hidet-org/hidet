from typing import Dict, List
from hidet.ir.expr import Var, TensorElement
from hidet.ir.stmt import BufferStoreStmt
from hidet.ir.func import Function, IRModule
from hidet.ir.dialects.compute import TensorNode
from hidet.ir.functors import collect, rewrite
from .base import Pass


class ApplyPrologueEpiloguePass(Pass):
    def process_module(self, ir_module: IRModule) -> IRModule:
        if len(ir_module.functions) != 1:
            raise ValueError('apply_prologue_epilogue_pass should run first before generate_packed_func pass.')
        func = list(ir_module.functions.values())[0]
        task = ir_module.task

        if not (len(func.params) == len(task.inputs) + len(task.outputs)):
            raise ValueError('The parameters of function should be the same as the sum of task inputs and outputs.')
        num_inputs = len(task.inputs)
        input2var: Dict[TensorNode, Var] = {a: b for a, b in zip(task.inputs, func.params[:num_inputs])}
        output2var: Dict[TensorNode, Var] = {a: b for a, b in zip(task.outputs, func.params[num_inputs:])}

        param_vars = [Var(param.name, param.data_type) for param in task.parameters]
        param2var = {a: b for a, b in zip(task.parameters, param_vars)}

        body = func.body

        # apply prologues
        for input_node, input_var in input2var.items():
            if input_node not in task.prologues:
                continue
            prologue = task.prologues[input_node]

            # the following collect assumes that there is no nested tensor elements for the same tensor, such as A[A[1, 2], 3]
            tensor_elements: List[TensorElement] = collect(body, TensorElement, stop_when_found=True)
            prologue_rewrite_map = {}
            for te in tensor_elements:
                if te.base is not input_var:
                    continue
                rmap = {}
                for extra_input in prologue.extra_inputs:
                    if extra_input not in param2var:
                        raise ValueError('Prologue used tensor {} that has not defined in task parameters.'.format(extra_input))
                    rmap[extra_input] = param2var[extra_input]
                for index_var, index_value in zip(prologue.indices, te.indices):
                    rmap[index_var] = index_value
                prologue_expr = rewrite(prologue.value, rmap)
                prologue_rewrite_map[te] = prologue_expr
            body = rewrite(body, prologue_rewrite_map)

        # apply epilogues
        for output_node, output_var in output2var.items():
            if output_node not in task.epilogues:
                continue
            epilogue = task.epilogues[output_node]

            # first check the usage of output var in TensorElement
            tensor_elements: List[TensorElement] = collect(body, TensorElement, stop_when_found=True)
            if any(te.base is output_var for te in tensor_elements):
                raise NotImplementedError('Currently do not support read from output tensor.')

            buffer_stores: List[TensorElement] = collect(body, BufferStoreStmt, stop_when_found=True)
            epilogue_rewrite_map = {}
            for bs in buffer_stores:
                if bs.base is not output_var:
                    continue
                rmap = {}
                for extra_input in epilogue.extra_inputs:
                    if extra_input not in param2var:
                        raise ValueError('Epilogue used tensor {} that has not defined in task parameters.'.format(extra_input))
                    rmap[extra_input] = param2var[extra_input]
                for index_var, index_value in zip(epilogue.indices, bs.indices):
                    rmap[index_var] = index_value
                epilogue_expr = rewrite(epilogue.value, rmap)
                if epilogue.out_indices and epilogue.out_tensor:
                    out_index_exprs = [rewrite(out_index_expr, rmap) for out_index_expr in epilogue.out_indices]
                    if epilogue.out_tensor not in param2var:
                        raise ValueError('Epilogue used a output tensor that has not defined in task parameters.'.format(epilogue.out_tensor))
                    out_tensor = param2var[epilogue.out_tensor]
                    epilogue_rewrite_map[bs] = BufferStoreStmt(out_tensor, out_index_exprs, epilogue_expr)
                else:
                    epilogue_rewrite_map[bs] = BufferStoreStmt(bs.base, bs.indices, epilogue_expr)
            body = rewrite(body, epilogue_rewrite_map)

        if body is func.body:
            return ir_module
        else:
            func = Function(func.name, params=param_vars, body=body, ret_type=func.ret_type, kind=func.kind,
                            local_vars=func.local_vars, local_const_vars=func.local_const_vars, extern_vars=func.extern_vars, attrs=func.attrs)
            ir_module = IRModule(funcs={func.name: func}, task=ir_module.task, global_vars=ir_module.global_vars)
            return ir_module


def apply_prologue_epilogue_pass() -> Pass:
    return ApplyPrologueEpiloguePass()
