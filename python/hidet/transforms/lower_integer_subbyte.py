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
# pylint: disable=unused-variable
from typing import Dict, List, Union, Tuple

from hidet.ir.tools import infer_type, simplify
from hidet.ir.type import BaseType, DataType, TensorType, TensorPointerType, PointerType
from hidet.ir.dtypes import i32
from hidet.ir.expr import Var, Expr, Add, TensorElement, Address, Constant, Cast, var, cast, bitwise_not
from hidet.ir.stmt import (
    Stmt,
    DeclareStmt,
    AssignStmt,
    LetStmt,
    EvaluateStmt,
    BufferStoreStmt,
    SeqStmt,
    BlackBoxStmt,
    WhileStmt,
    DeclareScope,
)

from hidet.ir.func import Function
from hidet.ir.module import IRModule
from hidet.ir.functors import IRRewriter
from hidet.transforms import Pass


def is_pointer_type(base_ty: BaseType):
    return isinstance(base_ty, (PointerType, TensorPointerType, TensorType))


def get_pointer_base_type(base_ty: BaseType):
    if isinstance(base_ty, PointerType):
        return base_ty.base_type
    elif isinstance(base_ty, TensorType):
        return base_ty.dtype
    else:
        ttype = base_ty.tensor_type
        return ttype.dtype


def is_integer_subbyte(dtype: BaseType):
    return dtype.is_data_type() and dtype.is_integer_subbyte()


class LowerIntegerSubbyteRewriter(IRRewriter):
    # convert subbyte integers to their storage type
    # e.g.,
    # a = register_tensor("int4b", [16]) ==> a = register_tensor("uint8", [8])
    # int4b* ptr = &a[8]                 ==> uint8* ptr = &a[4]
    # int4b* ptr = ptr + 10              ==> uint8* ptr = ptr + 5
    # we support tensor element access for global|shared|register tensors, and
    # we support buffer store statment for register tensors, i.e.
    # a = register_tensor("int4b", [4, 4])
    # b = a[2, 2]
    # a[2, 2] = int4b(-5)
    # user has to explicitly insert dtype conversion before applying arithmetic
    # operations on subbyte integers.
    # int4b a = int4b(-2)
    # int4b b = int4b(-3)
    # int4b c = a + b                    ==> not allowed
    def __init__(self):
        super().__init__()
        self.old2new: Dict[Var, Var] = {}
        self.stmts: List[Stmt] = []
        self.var2scope: Dict[Var, DeclareScope] = {}
        self.recursive_depth = 0

    def auto_var(self, v: Var = None, hint: str = None, e: Expr = None):
        if v is not None:
            self.stmts.append(DeclareStmt(v))
            return v
        v_ty = infer_type(e)
        v = var(hint, v_ty)
        self.stmts.append(DeclareStmt(v, e))
        return v

    def append_stmt(self, stmt: Union[Stmt, Expr]):
        if isinstance(stmt, Expr):
            stmt = EvaluateStmt(stmt)
        self.stmts.append(stmt)

    def flush_stmts(self):
        stmts = self.stmts
        self.stmts = []
        return stmts

    def flatten_stmts(self, stmts: List[Stmt]):
        if len(stmts) == 1:
            return stmts[0]
        else:
            return SeqStmt(stmts)

    def _get_divisor(self, dtype: DataType):
        storage_ty = dtype.storage
        storage_bits = storage_ty.nbits
        dtype_bits = dtype.nbits
        divisor = storage_bits // dtype_bits
        return divisor

    def _get_subbyte_value(self, dtype: DataType, base: Var, offset: Expr):
        storage_ty = dtype.storage
        storage_bits = storage_ty.nbits
        dtype_bits = dtype.nbits
        divisor = storage_bits // dtype_bits
        if divisor & (divisor - 1) != 0:
            raise TypeError(f"data type not supported yet(got:{dtype})")
        idx = simplify(offset // divisor)
        offset_ = simplify(offset & (divisor - 1))
        mask = storage_ty.constant(dtype.bits_mask)
        return (base[idx] >> (offset_ * dtype_bits)) & mask

    def _set_subbyte_value(self, dtype: DataType, base: Var, offset: Expr, value: Expr):
        storage_ty = dtype.storage
        storage_bits = storage_ty.nbits
        dtype_bits = dtype.nbits
        divisor = storage_bits // dtype_bits
        if divisor & (divisor - 1) != 0:
            raise TypeError(f"data type not supported yet(got:{dtype})")
        idx = simplify(offset // divisor)
        offset_ = simplify(offset & (divisor - 1))
        value_ty = infer_type(value)
        assert value_ty == storage_ty
        mask = storage_ty.constant(dtype.bits_mask)
        item = self.auto_var(hint="item", e=value & mask)
        updated_mask = self.auto_var(hint="updated_mask", e=bitwise_not(mask << (offset_ * dtype_bits)))
        new_bits = self.auto_var(hint="new_bits", e=item << (offset_ * dtype_bits))

        from hidet.ir.dtypes import u32, u16

        if self.var2scope[base].is_memory():
            if not any(storage_ty is ty for ty in [i32, u32, u16]):
                raise NotImplementedError(
                    "writing subbyte data to memory requires the storage type must be"
                    " int32, uint32, or uint16 due to atomicCAS, but got({storage_ty})"
                )
            original = self.auto_var(hint="original", e=storage_ty.zero)
            updated = self.auto_var(hint="updated", e=storage_ty.zero)
            body = []
            body.append(AssignStmt(original, base[idx]))
            body.append(AssignStmt(updated, (original & updated_mask) | new_bits))
            body.append(BlackBoxStmt("atomicCAS({}, {}, {});", ~base[idx], original, updated))
            body = SeqStmt(body)
            self.stmts.append(WhileStmt(original == updated, body))
        else:
            assert self.var2scope[base].is_register()
            original = self.auto_var(hint="original", e=base[idx])
            updated = self.auto_var(hint="updated", e=(original & updated_mask) | new_bits)
            self.stmts.append(BufferStoreStmt(base, [idx], updated))

    def visit_DataType(self, t: DataType):
        if t.is_integer_subbyte():
            return t.storage
        else:
            return t

    def visit_TensorType(self, t: TensorType):
        from hidet.ir.layout import row_major

        if is_integer_subbyte(t.dtype):
            shape = list(self.visit(t.shape))
            assert len(shape) == 1
            dtype = t.dtype
            storage_ty = dtype.storage
            storage_bits = storage_ty.nbits
            dtype_bits = dtype.nbits
            divisor = storage_bits // dtype_bits
            shape[-1] = shape[-1] // divisor
            layout = row_major(*shape)
            return TensorType(storage_ty, shape, layout)
        return super().visit_TensorType(t)

    def visit_Var(self, v: Var):
        if v in self.old2new:
            return self.old2new[v]
        return super().visit_Var(v)

    def visit_Constant(self, e: Constant):
        if is_integer_subbyte(e.type):
            ty = self.visit(e.type)
            value = self.visit(e.value)
            dtype = e.type
            storage_ty = dtype.storage
            mask = storage_ty.constant(dtype.bits_mask)
            value = value & mask
            return Constant(value, ty)
        return super().visit_Constant(e)

    def visit_TensorElement(self, e: TensorElement):
        if isinstance(e.base, Var):
            base_ty = infer_type(e.base)
            if is_pointer_type(base_ty):
                dtype = get_pointer_base_type(base_ty)
                if is_integer_subbyte(dtype):
                    base = self.visit(e.base)
                    assert len(e.indices) == 1
                    offset = self.visit(e.indices[0])
                    return self._get_subbyte_value(dtype, base, offset)
        return super().visit_TensorElement(e)

    def visit_Address(self, e: Address):
        if isinstance(e.expr, TensorElement):
            base = e.expr.base
            if isinstance(base, Var):
                base_ty = infer_type(base)
                if is_pointer_type(base_ty):
                    dtype = get_pointer_base_type(base_ty)
                    if is_integer_subbyte(dtype):
                        storage_ty = dtype.storage
                        storage_bits = storage_ty.nbits
                        dtype_bits = dtype.nbits
                        divisor = storage_bits // dtype_bits
                        assert len(e.expr.indices) == 1
                        offset = self.visit(e.expr.indices[0])
                        idx = simplify(offset // divisor)
                        base = self.visit(base)
                        return ~base[idx]
        return super().visit_Address(e)

    def _cast_int(self, dtype: DataType, expr: Expr):
        if not dtype.signedness():
            return expr
        int_type = i32
        int_data = cast(expr, int_type)
        shift = int_type.nbits - dtype.nbits
        return (int_data << shift) >> shift

    def visit_Cast(self, e: Cast):
        expr_ty = infer_type(e.expr)
        if is_integer_subbyte(expr_ty):
            if is_integer_subbyte(e.target_type):
                raise NotImplementedError(f"casting from {expr_ty} to {e.target_type} is not supported yet")
            expr = self.visit(e.expr)
            return cast(self._cast_int(expr_ty, expr), e.target_type)
        elif is_integer_subbyte(e.target_type):
            from hidet.ir.expr import if_then_else

            expr = self.visit(e.expr)
            dtype = e.target_type
            min_val = expr_ty(dtype.min_value.value)
            max_val = expr_ty(dtype.max_value.value)
            expr = if_then_else(expr < min_val, min_val, expr)
            expr = if_then_else(expr >= max_val, max_val, expr)
            if not dtype.signedness():
                return cast(expr, dtype.storage)
            storage_ty = dtype.storage
            int_type = i32
            int_data = cast(expr, int_type)
            shift = int_type.nbits - dtype.nbits
            int_data = (int_data << shift) >> shift
            return cast(int_data, storage_ty)
        target_type = self.visit(e.target_type)
        expr = self.visit(e.expr)
        if target_type is e.target_type and expr is e.expr:
            return e
        else:
            return cast(expr, target_type)

    def _subbyte_pointer_add(self, dtype: DataType, ptr: Union[Expr, Tuple[Expr]], offset: Expr):
        divisor = self._get_divisor(dtype)
        if isinstance(ptr, tuple):
            ptr, offset_ = ptr
            offset = offset + offset_
        if self.recursive_depth == 0:
            return ptr + offset // divisor
        else:
            return ptr, offset

    def visit_Add(self, e: Add):
        a_ty = infer_type(e.a)
        b_ty = infer_type(e.b)
        if isinstance(a_ty, PointerType) and is_integer_subbyte(a_ty.base_type):
            self.recursive_depth += 1
            a = self.visit(e.a)
            b = self.visit(e.b)
            self.recursive_depth -= 1
            return self._subbyte_pointer_add(a_ty.base_type, a, b)
        elif isinstance(b_ty, PointerType) and is_integer_subbyte(b_ty.base_type):
            self.recursive_depth += 1
            a = self.visit(e.a)
            b = self.visit(e.b)
            self.recursive_depth -= 1
            return self._subbyte_pointer_add(b_ty.base_type, b, a)
        return super().visit_Add(e)

    def visit_DeclareStmt(self, stmt: DeclareStmt):
        v_type = self.visit(stmt.var.type)
        if v_type is not stmt.var.type:
            v = var(stmt.var.hint, v_type)
            init = self.visit(stmt.init)
            self.old2new[stmt.var] = v
            if isinstance(v_type, TensorType):
                self.var2scope[v] = stmt.scope
            self.append_stmt(DeclareStmt(v, init, stmt.is_static, stmt.scope))
            return self.flatten_stmts(self.flush_stmts())
        self.append_stmt(super().visit_DeclareStmt(stmt))
        return self.flatten_stmts(self.flush_stmts())

    def visit_AssignStmt(self, stmt: AssignStmt):
        self.append_stmt(super().visit_AssignStmt(stmt))
        return self.flatten_stmts(self.flush_stmts())

    def visit_LetStmt(self, stmt: LetStmt):
        self.append_stmt(super().visit_LetStmt(stmt))
        return self.flatten_stmts(self.flush_stmts())

    def visit_BufferStoreStmt(self, stmt: BufferStoreStmt):
        if isinstance(stmt.buf, Var):
            buf_ty = infer_type(stmt.buf)
            if isinstance(buf_ty, TensorType):
                dtype = buf_ty.dtype
                if is_integer_subbyte(dtype):
                    buf = self.visit(stmt.buf)
                    indices = self.visit(stmt.indices)
                    value = self.visit(stmt.value)
                    assert len(indices) == 1
                    self._set_subbyte_value(dtype, buf, indices[0], value)
                    return self.flatten_stmts(self.flush_stmts())
        self.append_stmt(super().visit_BufferStoreStmt(stmt))
        return self.flatten_stmts(self.flush_stmts())


class LowerIntegerSubbytePass(Pass):
    def process_func(self, func: Function) -> Function:
        rewriter = LowerIntegerSubbyteRewriter()
        return rewriter(func)

    def process_module(self, ir_module: IRModule) -> IRModule:
        new_funcs = {}
        for name, func in ir_module.functions.items():
            new_funcs[name] = self.process_func(func)
        if all(new_funcs[name] is ir_module.functions[name] for name in new_funcs):
            return ir_module
        else:
            return ir_module.copy().reset_funcs(new_funcs, ir_module.global_vars)


def lower_integer_subbyte_pass() -> Pass:
    return LowerIntegerSubbytePass()
