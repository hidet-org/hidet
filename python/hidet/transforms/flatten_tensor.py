from typing import List, Union, Callable, Any
from hidet.ir.type import TensorType, tensor_type
from hidet.ir.expr import Var, TensorElement, TensorSlice, AlterLayout
from hidet.ir.stmt import BufferStoreStmt
from hidet.ir.func import Function
from hidet.ir.functors import simplify_to_int, FuncStmtExprRewriter
from hidet.ir.dialects.lowlevel import PointerType, TensorPointerType
from hidet.transforms import Pass
from hidet.ir.layout import StridesLayout, DataLayout


def concat_slices(lhs_indices, lhs_starts, lhs_ends, rhs_indices, rhs_starts=None, rhs_ends=None):
    if rhs_starts is None:
        rhs_starts = [None] * len(rhs_indices)
    if rhs_ends is None:
        rhs_ends = [None] * len(rhs_indices)
    assert len(lhs_indices) == len(lhs_starts) == len(lhs_ends)
    assert len(rhs_indices) == len(rhs_starts) == len(rhs_ends)
    indices = []
    starts = []
    ends = []
    i = 0
    for index, start, end in zip(lhs_indices, lhs_starts, lhs_ends):
        if index is not None:
            indices.append(index)
            starts.append(None)
            ends.append(None)
        else:
            assert i < len(rhs_indices)
            if rhs_indices[i] is not None:
                indices.append(start + rhs_indices[i] if start else rhs_indices[i])
                starts.append(None)
            elif rhs_starts[i] is not None:
                indices.append(None)
                starts.append(start + rhs_starts[i] if start else rhs_starts[i])
            else:
                indices.append(None)
                starts.append(None)
            # we ignore the end because we do not allow tensor-wise op.
            # end is only used for bound-checking, which is left in future.
            ends.append(None)
            i += 1
    assert i == len(rhs_indices)
    return indices, starts, ends


class FlattenTensorSliceRewriter(FuncStmtExprRewriter):
    # eliminate all TensorSlice
    # (A[:, 3])[2] will be converted to A[2, 3] and the slice op A[:, 3] will be eliminated.
    def visit_TensorSlice(self, e: TensorSlice):
        base = self.visit(e.base)
        if isinstance(base, TensorSlice):
            e_indices = [self.visit(i) if i else None for i in e.indices]
            e_starts = [self.visit(s) if s else None for s in e.starts]
            e_ends = [self.visit(e) if e else None for e in e.ends]
            indices, starts, ends = concat_slices(base.indices, base.starts, base.ends, e_indices, e_starts, e_ends)
            return TensorSlice(base.base, indices, starts, ends)
        else:
            return FuncStmtExprRewriter.visit_TensorSlice(self, e)

    def visit_TensorElement(self, e: TensorElement):
        base = self.visit(e.base)
        if isinstance(base, TensorSlice):
            e_indices = [self.visit(idx) for idx in e.indices]
            indices, starts, ends = concat_slices(base.indices, base.starts, base.ends, e_indices)
            assert not any(idx is None for idx in indices)
            return TensorElement(base.base, indices)
        else:
            return FuncStmtExprRewriter.visit_TensorElement(self, e)

    def visit_BufferStoreStmt(self, stmt: BufferStoreStmt):
        base = self.visit(stmt.buf)
        stmt_indices = [self.visit(idx) for idx in stmt.indices]
        if isinstance(base, TensorSlice):
            indices, starts, ends = concat_slices(base.indices, base.starts, base.ends, stmt_indices)
            assert not any(idx is None for idx in indices)
            return BufferStoreStmt(base.base, indices, self.visit(stmt.value))
        else:
            return FuncStmtExprRewriter.visit_BufferStoreStmt(self, stmt)


class FlattenTensorAccessRewriter(FuncStmtExprRewriter):
    # flatten all high-dimension tensor access
    # A = int[3, 4]
    #   TensorElement:  A[2, 1]     ==> A[2 * 4 + 1]
    # BufferStoreStmt:  A[2, 1] = 3 ==> A[2 * 4 + 1] = 3
    def visit_Function(self, func: Function):
        for var in func.params + func.local_vars:
            if isinstance(var.type, TensorType):
                size = simplify_to_int(var.type.layout.size)
                self.memo[var] = Var(var.hint, tensor_type(var.type.scope, var.type.scalar_type, [size], DataLayout.row_major([size])))
            elif isinstance(var.type, TensorPointerType):
                self.memo[var] = var
        body = self(func.body)
        params = [self(p) for p in func.params]
        local_vars = [self(v) for v in func.local_vars]
        return Function(func.name, params, body, func.ret_type, local_vars, func.extern_vars, func.attrs)

    @staticmethod
    def get_layout(e) -> Callable[..., Any]:
        if isinstance(e, AlterLayout):
            return lambda *indices: FlattenTensorAccessRewriter.get_layout(e.var)(e.layout_map(indices))
        elif isinstance(e, Var):
            if isinstance(e.type, TensorType):
                return e.type.layout
            elif isinstance(e.type, TensorPointerType):
                return e.type.tensor_type.layout
            elif isinstance(e.type, PointerType):
                return StridesLayout(shape=[0], strides=[1])
        raise ValueError("Can not infer layout from '{}'".format(type(e)))

    def visit_TensorElement(self, e: TensorElement):
        var = self(e.base)
        indices = [self(i) for i in e.indices]
        layout = self.get_layout(e.base)
        global_index = layout(*indices)
        return TensorElement(var, [global_index])

    def visit_BufferStoreStmt(self, stmt: BufferStoreStmt):
        var = self(stmt.buf)
        indices = [self(i) for i in stmt.indices]
        value = self(stmt.value)
        layout = self.get_layout(stmt.buf)
        global_index = layout(indices)
        return BufferStoreStmt(var, [global_index], value)

    def visit_AlterLayout(self, e: AlterLayout):
        return self(e.var)

    def visit_TensorSlice(self, e: TensorSlice):
        raise ValueError('there should not be any tensor slice after flattening tensor slice.')


class FlattenTensorPass(Pass):
    def process_func(self, func: Function) -> Function:
        flatten_slice = FlattenTensorSliceRewriter()
        flatten_access = FlattenTensorAccessRewriter()
        return flatten_access(flatten_slice(func))


def flatten_tensor_pass():
    return FlattenTensorPass()
