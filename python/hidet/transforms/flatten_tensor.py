import functools
import operator
from typing import List
from hidet.ir.type import TensorType
from hidet.ir.expr import Var, TensorElement
from hidet.ir.stmt import BufferStoreStmt
from hidet.ir.func import Function
from hidet.ir.functors import collect, rewrite, simplify, simplify_to_int
from hidet.ir.dialects.lowlevel import PointerType, Address, TensorPointerType
from hidet.transforms import Pass
from hidet.ir.layout import StridesLayout, DataLayout


class FlattenTensorPass(Pass):
    def process_func(self, func: Function):
        # we do not flatten tensors with the shared memory and register scope tensors
        target_tensors = []
        flattened_vars = []
        var2tensor_type = {}
        for var in func.params + func.local_vars:
            if isinstance(var.type, (TensorPointerType, TensorType)):
                target_tensors.append(var)
                if isinstance(var.type, TensorType):
                    assert isinstance(var.type.layout, (DataLayout, StridesLayout))
                    shape = [simplify_to_int(var.type.layout.size)]
                    var2tensor_type[var] = var.type
                    flattened_vars.append(Var(var.hint, TensorType(var.type.scope, var.type.scalar_type, shape, [1])))
                else:
                    var2tensor_type[var] = var.type.tensor_type
                    flattened_vars.append(var)

        if len(target_tensors) == 0:
            return func
        tensor2flattened = {t: f for t, f in zip(target_tensors, flattened_vars)}

        body = func.body
        rmap = {}
        # update TensorElement
        element_exprs: List[TensorElement] = collect(body, TensorElement)
        for e in element_exprs:
            if e.base not in target_tensors:
                continue
            tensor_var: Var = e.base
            tensor_type = var2tensor_type[tensor_var]
            global_index = tensor_type.layout(*e.indices)
            rmap[e] = TensorElement(tensor2flattened[tensor_var], [global_index])
        body = rewrite(body, rmap)

        # update BufferStoreStmt
        rmap.clear()
        store_stmts: List[BufferStoreStmt] = collect(body, BufferStoreStmt)
        for s in store_stmts:
            if s.buf not in target_tensors:
                continue
            tensor_var: Var = s.buf
            tensor_type = var2tensor_type[tensor_var]
            global_index = tensor_type.layout(*s.indices)
            rmap[s] = BufferStoreStmt(tensor2flattened[tensor_var], [global_index], s.value)
        body = rewrite(body, rmap)

        # update other usage of original Tensor var to Pointer var
        rmap = tensor2flattened
        body = rewrite(body, rmap)

        params = [tensor2flattened[p] if p in tensor2flattened else p for p in func.params]
        local_vars = [tensor2flattened[v] if v in tensor2flattened else v for v in func.local_vars]
        new_func = Function(func.name, params, body, func.ret_type, local_vars, func.attrs)
        return new_func


def flatten_tensor_pass():
    return FlattenTensorPass()
