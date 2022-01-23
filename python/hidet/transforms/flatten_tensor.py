import functools
import operator
from typing import List
from hidet.ir.type import TensorType
from hidet.ir.expr import Var, TensorElement, TensorSlice
from hidet.ir.stmt import BufferStoreStmt
from hidet.ir.func import Function
from hidet.ir.functors import collect, rewrite, simplify
from hidet.ir.dialects.lowlevel import PointerType, Address
from hidet.transforms import Pass
from hidet.ir.type import StridesLayout, LocalLayout


class FlattenTensor(Pass):
    def __init__(self):
        super().__init__('flatten_tensor')

    def process_func(self, func: Function):
        # we do not flatten tensors with the shared memory and register scope tensors
        target_tensors = []
        flattened_vars = []
        for param in func.params:
            if isinstance(param.type, TensorType):
                target_tensors.append(param)
                flattened_vars.append(Var(param.hint, PointerType(param.type.scalar_type)))
        for local_var in func.local_vars:
            if isinstance(local_var.type, TensorType):
                target_tensors.append(local_var)
                scope = local_var.type.scope
                dtype = local_var.type.scalar_type
                shape = [int(simplify(functools.reduce(operator.mul, local_var.type.shape)))]
                flattened_vars.append(Var(local_var, TensorType(scope, dtype, shape, [1])))

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
            assert isinstance(tensor_var.type, TensorType)
            global_index = tensor_var.type.layout.serialize(*e.indices)
            rmap[e] = TensorElement(tensor2flattened[tensor_var], [global_index])
        body = rewrite(body, rmap)

        # update TensorSlice
        rmap.clear()
        slice_exprs: List[TensorSlice] = collect(func.body, TensorSlice)
        assert len(slice_exprs) == 0 # deprecated, will remove
        # for e in slice_exprs:
        #     if e.base not in target_tensors:
        #         continue
        #     tensor_var: Var = e.base
        #     assert isinstance(tensor_var.type, TensorType)
        #     shape = tensor_var.type.shape
        #     strides = tensor_var.type.layout
        #
        #     indices = []
        #     start_idx = 0
        #     items = []
        #     for i in range(len(e.indices)):
        #         if e.indices[i]:
        #             indices.append(e.indices[i])
        #         else:
        #             indices.append(e.starts[start_idx])
        #             start_idx += 1
        #         items.append(indices[i] * strides[i])
        #     global_index = sum(items)
        #     rmap[e] = Address(TensorElement(tensor2flattened[tensor_var], [global_index]))
        # body = rewrite(body, rmap)

        # update BufferStoreStmt
        rmap.clear()
        store_stmts: List[BufferStoreStmt] = collect(body, BufferStoreStmt)
        for s in store_stmts:
            if s.buf not in target_tensors:
                continue
            tensor_var: Var = s.buf
            assert isinstance(tensor_var.type, TensorType)
            global_index = tensor_var.type.layout.serialize(*s.indices)
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
    return FlattenTensor()
