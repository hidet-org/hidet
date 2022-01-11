from typing import List
from hidet.ir.type import TensorType
from hidet.ir.expr import Var, TensorElement, TensorSlice
from hidet.ir.stmt import BufferStoreStmt
from hidet.ir.func import Function
from hidet.ir.functors import collect, rewrite
from hidet.ir.dialects.lowlevel import PointerType, Address
from hidet.transforms import Pass


def sum(lst: List):
    assert len(lst) > 0
    if len(lst) == 1:
        return lst[0]
    else:
        rt = lst[0] + lst[1]
        for v in lst[2:]:
            rt = rt + v
        return rt


class FlattenTensor(Pass):
    def __init__(self):
        super().__init__('flatten_tensor')

    def process_func(self, func: Function):
        params = func.params
        # we do not flatten tensors with the shared memory and register scope tensors
        target_params = [p for p in params if isinstance(p.type, TensorType)
                         and p.type.scope.name in ['global', 'host']]
        if len(target_params) == 0:
            return func
        pointer_params = [Var(p.hint, PointerType(p.type.scalar_type)) for p in target_params]
        global2pointer = {g: p for g, p in zip(target_params, pointer_params)}

        rmap = {}
        # update BufferStoreStmt
        store_stmts: List[BufferStoreStmt] = collect(func.body, BufferStoreStmt)
        for s in store_stmts:
            if s.buf not in target_params:
                continue
            param: Var = s.buf
            assert isinstance(param.type, TensorType)
            shape = param.type.shape
            strides = param.type.strides
            indices = s.indices
            items = []
            for i in range(len(shape)):
                items.append(indices[i] * strides[i])
            global_index = sum(items)
            rmap[s] = BufferStoreStmt(global2pointer[param], [global_index], s.value)

        # update TensorElement
        element_exprs: List[TensorElement] = collect(func.body, TensorElement)
        for e in element_exprs:
            if e.base not in target_params:
                continue
            param: Var = e.base
            assert isinstance(param.type, TensorType)
            shape = param.type.shape
            strides = param.type.strides
            indices = e.indices
            items = []
            for i in range(len(shape)):
                items.append(indices[i] * strides[i])
            global_index = sum(items)
            rmap[e] = TensorElement(global2pointer[param], [global_index])

        # update TensorSlice
        slice_exprs: List[TensorSlice] = collect(func.body, TensorSlice)
        for e in slice_exprs:
            if e.base not in target_params:
                continue
            param: Var = e.base
            assert isinstance(param.type, TensorType)
            shape = param.type.shape
            strides = param.type.strides

            indices = []
            start_idx = 0
            items = []
            for i in range(len(e.indices)):
                if e.indices[i]:
                    indices.append(e.indices[i])
                else:
                    indices.append(e.starts[start_idx])
                    start_idx += 1
                items.append(indices[i] * strides[i])
            global_index = sum(items)
            rmap[e] = Address(TensorElement(global2pointer[param], [global_index]))

        # update other usage of original Tensor var to Pointer var
        rmap.update(global2pointer)

        new_func = Function(func.name, None, None, func.ret_type, func.local_vars, func.attrs)
        new_func.params = [global2pointer[p] if p in global2pointer else p for p in func.params]
        new_func.body = rewrite(func.body, rmap)
        return new_func


def flatten_tensor_pass():
    return FlattenTensor()
