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
from typing import List, Dict, Union, Optional, Callable

from hidet.ir.type import DataType, PointerType
from hidet.ir.expr import Var, Expr, var
from hidet.ir.stmt import DeclareScope
from hidet.ir.tools import infer_type
from hidet.ir.functors import IRVisitor, IRRewriter

from hidet.ir.cute.expr import CallOp, CConst
from hidet.ir.cute.type import TiledTensorType

from hidet.ir.func import Function
from hidet.transforms.base import FunctionPass
from hidet.ir.stmt import DeclareStmt

from hidet.ir.primitives import ldg128, ldg64, ldg32, stg128, stg64, stg32, lds128, lds64, lds32, sts128, sts64, sts32
from hidet.ir.primitives.cuda.mma import ldmatrix
from hidet.ir.primitives.cuda.cp_async import cp_async

from hidet.ir.cute.ops import Copy, Mask
from hidet.ir.cute.algorithm import TiledCopy
from hidet.ir.cute.layout import (
    TiledTensorLayout,
    ComposedTensorLayout,
    TensorLayout,
    composition,
    make_layout,
    max_common_vector,
    left_inverse,
    coalesce,
    filter,
)
from hidet.ir.cute.int_tuple import rank, compact_col_major, flatten, depth, shape_div
from hidet.utils import initialize
from .lower_ops import Buffer


# The bytes of a general purpose register in CUDA is 4
BYTES_PER_GPR = 4
BITS_PER_BYTE = 8


def group(layout: Union[TensorLayout, ComposedTensorLayout], size: int):
    flat_shape = flatten(layout.shape)
    flat_stride = flatten(layout.stride)
    flat_shape = flat_shape if isinstance(flat_shape, tuple) else [flat_shape]
    flat_stride = flat_stride if isinstance(flat_stride, tuple) else [flat_stride]
    result_shape = []
    result_stride = []
    rest_shape = []
    rest_stride = []
    current_idx = 1
    rest = False
    for s, d in zip(flat_shape, flat_stride):
        if d == 0:
            if rest:
                rest_shape.append(s)
                rest_stride.append(d)
            else:
                result_shape.append(s)
                result_stride.append(d)
        elif current_idx * s <= size:
            result_shape.append(s)
            result_stride.append(d)
            current_idx *= s
            if current_idx == size:
                rest = True
        elif current_idx * s > size:
            if not rest:
                shape = shape_div(size, current_idx)
                remaining = shape_div(s, shape)
                result_shape.append(shape)
                result_stride.append(d)
                rest_shape.append(remaining)
                rest_stride.append(d * shape)
                rest = True
            else:
                rest_shape.append(s)
                rest_stride.append(d)
            current_idx *= s

    def tensor_layout(shape, stride):
        if len(shape) > 1:
            return TensorLayout(tuple(shape), tuple(stride))
        elif len(shape) == 1:
            return TensorLayout(shape[0], stride[0])
        else:
            return TensorLayout(1)

    result = tensor_layout(result_shape, result_stride)
    rest = filter(tensor_layout(rest_shape, rest_stride), False)
    if isinstance(layout, TensorLayout):
        return result, rest
    else:
        assert isinstance(layout, ComposedTensorLayout)
        return ComposedTensorLayout(result, layout.base, layout.functor), ComposedTensorLayout(
            rest, layout.base, layout.functor
        )


class CopyInstruction:
    def __init__(
        self,
        apply: Union[Callable, None],
        shape,
        src_scope: Union[DeclareScope, str],
        dst_scope: Union[DeclareScope, str],
        src_layout: TensorLayout,
        dst_layout: TensorLayout,
    ):
        self.apply: Callable = apply
        self.shape = shape
        self.src_scope = src_scope if isinstance(src_scope, DeclareScope) else DeclareScope.from_str(src_scope)
        self.dst_scope = dst_scope if isinstance(dst_scope, DeclareScope) else DeclareScope.from_str(dst_scope)
        self.src_layout: TensorLayout = src_layout
        self.dst_layout: TensorLayout = dst_layout
        self.bytes_per_inst = self.dst_layout[1].size() // BITS_PER_BYTE

    def logical_match(self, src: Buffer, dst: Buffer, tiled_copy: TiledCopy):
        src_tv_layout_inst = self._get_layout_in_element(src.buffer, self.src_layout)
        dst_tv_layout_inst = self._get_layout_in_element(dst.buffer, self.dst_layout)
        src_thr_layout_inst, src_val_layout_inst = (src_tv_layout_inst[0], src_tv_layout_inst[1])
        dst_thr_layout_inst, dst_val_layout_inst = (dst_tv_layout_inst[0], dst_tv_layout_inst[1])

        _, src_tv_layout = tiled_copy.src_tv_layout()
        _, dst_tv_layout = tiled_copy.dst_tv_layout()
        src_thr_layout, src_val_layout = src_tv_layout[0][0], coalesce(
            make_layout(src_tv_layout[0][1], src_tv_layout[1])
        )
        dst_thr_layout, dst_val_layout = dst_tv_layout[0][0], coalesce(
            make_layout(dst_tv_layout[0][1], dst_tv_layout[1])
        )

        src_thr_inst, _ = group(src_thr_layout, filter(src_thr_layout_inst).size())
        src_val_inst, _ = group(src_val_layout, src_val_layout_inst.size())
        dst_thr_inst, _ = group(dst_thr_layout, dst_thr_layout_inst.size())
        dst_val_inst, _ = group(dst_val_layout, dst_val_layout_inst.size())

        # calculate the conversion mapping
        cvt_src = coalesce(
            composition(make_layout(src_thr_inst, src_val_inst), left_inverse(filter(src_tv_layout_inst)))
        )
        cvt_dst = coalesce(composition(make_layout(dst_thr_inst, dst_val_inst), left_inverse(dst_tv_layout_inst)))

        # print(f"src_thr_layout:{src_thr_layout}")
        # print(f"src_val_layout:{src_val_layout}")
        # print(f"dst_thr_layout:{dst_thr_layout}")
        # print(f"dst_val_layout:{dst_val_layout}")

        # print(f"dst_thr_layout_inst:{dst_thr_layout_inst}")
        # print(f"dst_val_layout_inst:{dst_val_layout_inst}")

        # print(f"{cvt_src}, {cvt_dst}")

        return cvt_src == cvt_dst

    def inst_match(self, src: Buffer, dst: Buffer):
        if src.scope != self.src_scope or dst.scope != self.dst_scope:
            return None
        src_tv_layout_inst = self._get_layout_in_element(src.buffer, self.src_layout)
        dst_tv_layout_inst = self._get_layout_in_element(dst.buffer, self.dst_layout)
        src_val_layout_inst = src_tv_layout_inst[1]
        dst_val_layout_inst = dst_tv_layout_inst[1]

        src_val_inst, src_val_rest = group(src.layout, src_val_layout_inst.size())
        dst_val_inst, dst_val_rest = group(dst.layout, dst_val_layout_inst.size())
        if self.src_scope.is_memory() and filter(src_val_inst).shape != src_val_layout_inst.shape:
            return None
        if self.dst_scope.is_memory() and filter(dst_val_inst).shape != dst_val_layout_inst.shape:
            return None
        return make_layout(src_val_inst, src_val_rest), make_layout(dst_val_inst, dst_val_rest)

    def match(self, src: Buffer, dst: Buffer, tiled_copy: TiledCopy = None):
        raise NotImplementedError()

    def _get_element_bytes(self, e: Expr):
        ty = infer_type(e)
        assert isinstance(ty, (TiledTensorType, PointerType))
        if isinstance(ty, TiledTensorType):
            return ty.dtype.nbytes
        else:
            return ty.base_type.nbytes

    def _get_register_pointers(self, e: Expr):
        element_bytes = self._get_element_bytes(e)
        assert element_bytes <= BYTES_PER_GPR
        incr = BYTES_PER_GPR // element_bytes
        pointers: List[Expr] = []
        for inc in range(0, self.bytes_per_inst // element_bytes, incr):
            pointers.append(e + inc)
        return pointers

    def _get_layout_in_element(self, e: Expr, layout: TensorLayout):
        element_bits = self._get_element_bytes(e) * BITS_PER_BYTE
        assert depth(layout.shape) == 2 and depth(layout.stride) == 2
        thr_rank = rank(layout[0].stride)
        val_rank = rank(layout[1].stride)
        flat_shape = flatten(layout.shape)
        flat_stride = flatten(layout.stride)
        index = range(len(flat_stride))
        sorted_dsi = sorted(zip(flat_stride, flat_shape, index))
        shape = [s // element_bits if d == self.shape[0] and (s % element_bits == 0) else s for d, s, _ in sorted_dsi]
        stride = []
        for s, (d, _, _) in zip(shape, sorted_dsi):
            if d != 0:
                stride.append(s)
        stride = [0] * (len(shape) - len(stride)) + list(compact_col_major(tuple(stride)))
        index = [i for _, _, i in sorted_dsi]
        result_shape = [0] * (thr_rank + val_rank)
        result_stride = [0] * (thr_rank + val_rank)
        for s, d, i in zip(shape, stride, index):
            result_shape[i] = s
            result_stride[i] = d
        shapes = result_shape[:thr_rank], result_shape[thr_rank:]
        strides = result_stride[:thr_rank], result_stride[thr_rank:]
        layouts = [coalesce(TensorLayout(tuple(shape), tuple(stride))) for shape, stride in zip(shapes, strides)]
        return make_layout(*layouts)

    def __call__(self, src: Expr, dst: Expr, mask: Optional[Expr] = None):
        raise NotImplementedError()


class UniversalCopyInstruction(CopyInstruction):
    def __init__(
        self,
        apply: Callable,
        shape,
        src_scope: DeclareScope,
        dst_scope: DeclareScope,
        src_layout: TensorLayout,
        dst_layout: TensorLayout = None,
        require_mask: bool = False,
    ):
        dst_layout = src_layout if dst_layout is None else dst_layout
        super().__init__(apply, shape, src_scope, dst_scope, src_layout, dst_layout)
        self.require_mask = require_mask

    def match(self, src: Buffer, dst: Buffer, tiled_copy: TiledCopy = None):
        if self.bytes_per_inst > max_common_vector(src.layout, dst.layout) * self._get_element_bytes(src.buffer):
            return None
        if tiled_copy is None or super().logical_match(src, dst, tiled_copy):
            return super().inst_match(src, dst)
        return None

    def __call__(self, src: Expr, dst: Expr, mask: Optional[Expr] = None):
        from hidet.ir.stmt import IfStmt, AssignStmt

        # fallback case
        if self.bytes_per_inst < 4:
            if self.require_mask:
                return IfStmt(mask, AssignStmt(dst[0], src[0]))
            else:
                return AssignStmt(dst[0], src[0])
        operands: List[Expr] = []
        if self.src_scope == DeclareScope.Register:
            operands.extend(self._get_register_pointers(src))
        if self.dst_scope == DeclareScope.Register:
            operands.extend(self._get_register_pointers(dst))
        if self.src_scope != DeclareScope.Register:
            operands.append(src)
        if self.dst_scope != DeclareScope.Register:
            operands.append(dst)
        if mask is not None:
            assert self.require_mask
            operands.append(mask)
        elif self.require_mask:
            operands.append(True)
        return self.apply(*operands)


class AsyncCopyInstruction(UniversalCopyInstruction):
    def __init__(
        self,
        apply: Callable,
        shape,
        src_scope: DeclareScope,
        dst_scope: DeclareScope,
        src_layout: TensorLayout,
        dst_layout: TensorLayout = None,
    ):
        super().__init__(apply, shape, src_scope, dst_scope, src_layout, dst_layout)

    def __call__(self, src: Expr, dst: Expr, mask: Optional[Expr] = None):
        operands: List[Expr] = []
        assert self.src_scope.is_global() and self.dst_scope.is_shared()
        operands.append(dst)
        operands.append(src)
        operands.append(self.bytes_per_inst)
        cache_level = "global" if self.bytes_per_inst == 16 else "always"
        return self.apply(*operands, src_size=None, cache_level=cache_level)


class LdMatrix(CopyInstruction):
    def __init__(
        self,
        apply: Callable,
        shape,
        src_scope: DeclareScope,
        dst_scope: DeclareScope,
        src_layout: TensorLayout,
        dst_layout: TensorLayout,
        trans: bool = False,
    ):
        super().__init__(apply, shape, src_scope, dst_scope, src_layout, dst_layout)
        self.trans = trans

    def match(self, src: Buffer, dst: Buffer, tiled_copy: TiledCopy = None):
        if tiled_copy is None:
            return None
        elif super().logical_match(src, dst, tiled_copy):
            return super().inst_match(src, dst)
        else:
            return None

    def __call__(self, src: Expr, dst: Expr, mask: Optional[Expr] = None):
        assert mask is None
        assert self.src_scope == DeclareScope.Shared and self.dst_scope == DeclareScope.Register
        from hidet.ir.dtypes import u32
        from hidet.ir.expr import deref, cast

        reg_ptrs = self._get_register_pointers(dst)
        regs = [deref(cast(ptr, ~u32)) for ptr in reg_ptrs]
        return self.apply(regs, src, shared_space_addr=False, trans=self.trans)


memory_instructions = []


@initialize()
def register_universal_copy_instruction():
    # remove ldg256 because instructions wider than 16-bytes do not boost
    # performnce
    # insts = [(256, ldg256), (128, ldg128), (64, ldg64), (32, ldg32), (16, None)]
    insts = [(128, ldg128), (64, ldg64), (32, ldg32), (16, None)]
    for load_bits, inst in insts:
        shape = (1, load_bits)
        src_layout = TensorLayout(((1), (1, load_bits)), ((1), (1, 1)))
        memory_instructions.append(
            UniversalCopyInstruction(inst, shape, "global", "register", src_layout, require_mask=True)
        )

    insts = [(128, cp_async), (64, cp_async), (32, cp_async)]
    for load_bits, inst in insts:
        shape = (1, load_bits)
        src_layout = TensorLayout(((1), (1, load_bits)), ((1), (1, 1)))
        memory_instructions.append(AsyncCopyInstruction(inst, shape, "global", "shared", src_layout))

    insts = [(128, lds128), (64, lds64), (32, lds32), (16, None)]
    for load_bits, inst in insts:
        shape = (1, load_bits)
        src_layout = TensorLayout(((1), (1, load_bits)), ((1), (1, 1)))
        memory_instructions.append(UniversalCopyInstruction(inst, shape, "shared", "register", src_layout))

    # remove stg512, stg256 because instructions wider than 16-bytes do not boost
    # performnce
    # insts = [(512, stg512), (256, stg256), (128, stg128), (64, stg64), (32, stg32), (16, None)]
    insts = [(128, stg128), (64, stg64), (32, stg32), (16, None)]
    for store_bits, inst in insts:
        shape = (1, store_bits)
        src_layout = TensorLayout(((1), (1, store_bits)), ((1), (1, 1)))
        memory_instructions.append(
            UniversalCopyInstruction(inst, shape, "register", "global", src_layout, require_mask=True)
        )

    insts = [(128, sts128), (64, sts64), (32, sts32), (16, None)]
    for store_bits, inst in insts:
        shape = (1, store_bits)
        src_layout = TensorLayout(((1), (1, store_bits)), ((1), (1, 1)))
        memory_instructions.append(UniversalCopyInstruction(inst, shape, "register", "shared", src_layout))


@initialize()
def register_ldmatrix():
    for nr_mat in [1, 2, 4]:
        for trans in [True, False]:
            shape = (8 * nr_mat, 128)
            src_layout = TensorLayout(((8 * nr_mat, 32 // (8 * nr_mat)), (128)), ((1, 0), (8 * nr_mat)))
            if trans:
                dst_layout = TensorLayout(((4, 8), (2, 16, nr_mat)), ((2, 128 * nr_mat), (1, 8 * nr_mat, 8)))
            else:
                dst_layout = TensorLayout(((4, 8), (32, nr_mat)), ((32 * nr_mat, 1), (8 * nr_mat, 8)))
            memory_instructions.append(LdMatrix(ldmatrix, shape, "shared", "register", src_layout, dst_layout, trans))


class CopyInstructionSelection(IRRewriter):
    def __init__(self):
        super().__init__()

    def visit_Copy(self, e: Copy):
        args = [self.visit(arg) for arg in e.args]

        def arg_to_buffer(arg: Expr):
            arg_ty: TiledTensorType = infer_type(arg)
            layout: Union[TiledTensorLayout, ComposedTensorLayout, TensorLayout] = arg_ty.layout
            dtype: DataType = arg_ty.dtype
            scope: DeclareScope = arg_ty.scope
            return Buffer(buffer=arg, offset=None, dtype=dtype, scope=scope, layout=layout)

        bufs = [arg_to_buffer(arg) for arg in args[:2]]
        candidates = []
        for inst in memory_instructions:
            result = inst.match(bufs[0], bufs[1], e.tiled_copy)
            if result is not None:
                candidates.append((inst, *result))
        if len(candidates) == 0:
            raise NotImplementedError(f"Cannot find a suitable instruction for {e}")
        candidates = sorted(candidates, key=lambda x: -x[0].bytes_per_inst)
        inst, src_layout, dst_layout = candidates[0]
        annotations: Dict[str, CConst] = {}
        annotations["inst"] = inst
        annotations["src_layout"] = src_layout
        annotations["dst_layout"] = dst_layout
        return e.reforward(args, annotations_update=annotations)


class MaskUsageMarker(IRVisitor):
    def __init__(self):
        super().__init__()
        self.mask2copy: Dict[Mask, Copy] = {}
        self.var2mask: Dict[Var, Mask] = {}
        self.var2copy: Dict[Var, Copy] = {}

    def mark(self, func: Function):
        self.visit(func)
        return self.mask2copy

    def visit_DeclareStmt(self, stmt: DeclareStmt):
        v = stmt.var
        if isinstance(stmt.init, CallOp):
            call = stmt.init
            op = call.op
            if isinstance(op, Mask):
                self.var2mask[v] = op
                if v in self.var2copy:
                    copy = self.var2copy[v]
                    self.mask2copy[op] = copy

    def visit_Copy(self, e: Copy):
        if e.mask is not None:
            if isinstance(e.mask, Var):
                v = e.mask
                self.var2copy[v] = e
                if v in self.var2mask:
                    mask = self.var2mask[v]
                    self.mask2copy[mask] = e
            else:
                assert isinstance(e.mask, CallOp)
                mask = e.mask.op
                self.mask2copy[mask] = e


class MaskAnnotation(IRRewriter):
    def __init__(self, mask2copy: Dict[Mask, Copy]):
        super().__init__()
        self.mask2copy = mask2copy
        self.old2new: Dict[Var, Var] = {}

    def visit_Copy(self, e: Copy):
        src = self.visit(e.src)
        dst = self.visit(e.dst)
        if e.mask is not None:
            assert e.mask in self.old2new
            mask = self.old2new[e.mask]
        else:
            mask = None
        if src is e.src and dst is e.dst and mask is e.mask:
            return e
        else:
            return e.reforward([src, dst, mask])

    def visit_DeclareStmt(self, stmt: DeclareStmt):
        v = self.visit(stmt.var)
        if isinstance(stmt.init, CallOp) and isinstance(stmt.init.op, Mask):
            call = stmt.init
            op = self.visit(call.op)
            init = op.make_call()
            v = var(v.hint, infer_type(init))
            self.old2new[stmt.var] = v
            return DeclareStmt(v, init, stmt.is_static, stmt.scope)
        else:
            init = self.visit(stmt.init) if stmt.init is not None else None
            if v is stmt.var and init is stmt.init:
                return stmt
            else:
                return DeclareStmt(v, init, stmt.is_static, stmt.scope)

    def visit_Mask(self, e: Mask):
        args = [self.visit(arg) for arg in e.args]
        if e not in self.mask2copy:
            raise ValueError("Operator mask{e} hasn't been marked, there should be some bugs.")
        copy = self.mask2copy[e]
        src_layout = copy.annotations["src_layout"]
        inst_len = src_layout[0].count()
        tiled_copy: TiledCopy = e.tiled_copy
        _, src_thrval_layout = tiled_copy.src_tv_layout()
        val_layout = coalesce(make_layout(src_thrval_layout[0][1], src_thrval_layout[1]))
        _, rest_layout = group(val_layout, inst_len)
        annotations: Dict[str, CConst] = {}
        annotations["rest_layout"] = rest_layout
        return e.reforward(args, annotations_update=annotations)


class InstructionSelectionPass(FunctionPass):
    """
    In the CuTE dialect, we propose a code generation algorithm that first selects the optimal instruction
    for each copy operation, and then generates loops that repeatedly execute the selected instructions.
    This pass aims to select the best instruction for each copy operation and annotate the mask for each
    copy operation accordingly.

    Example:
    Consider the following copy operation defined by the `src_tv_layout` and `dst_tv_layout`:
    ```python
    src_tv_layout = TensorLayout(((4, 8), (2, 2)), ((32, 1), (16, 8)))
    dst_tv_layout = TensorLayout(((4, 8), (2, 2)), ((32, 1), (16, 8)))
    copy_atom = CopyAtom('warp', (16, 8), src_tv_layout, dst_tv_layout)
    warp_in_thread_block = Level('warp', 'thread_block', (2, 2), TensorLayout((2, 2), (1, 2))) # Four
    warps organized as a 2x2 grid
    tiled_copy = TiledCopy(copy_atom, [warp_in_thread_block])
    ```

    Similarly, instructions can be encapsulated with TV layouts. For instance, the `ldmatrix` instruction
    can be represented as:
    ```python
    src_tv_layout = TensorLayout(((32,), (8, 4)), ((1,), (32, 0)))
    dst_tv_layout = TensorLayout(((4, 8), (2, 4)), ((64, 1), (32, 8)))
    ```

    An instruction can be used to perform the copy operation if the source and destination tensor layouts
    and the instruction's layouts satisfy the following condition:
    ```
    f o p^{-1} = g o q^{-1}
    ```
    where `f` and `g` denote the TV layouts for the copy operation, and `p` and `q` represent the TV layouts
    for the instruction.

    Using this condition, we pick the widest instruction that can be used for the copy operation.

    Once the instruction has been selected, we generate loops to perform the copy operation using the selected
    instruction. During the instruction selection pass, the TV layouts of the source and destination tensors
    will be grouped into two modes: `inst` and `rest`. Assuming the rest mode has the shape `(s1, s2, ...)`,
    the loop can be represented as:
    ```python
    for i1 in range(s1):
        for i2 in range(s2):
            ...
            inst(src[i1, i2, ...], dst[i1, i2, ...])
    ```

    Note: The code generation for mask operations depends on the selected instruction for the copy operation. Therefore,
    we need to annotate the mask with the selected instruction to ensure the generated code is correct.

    Methods:
        process_func(func: Function) -> Function:
            Processes the given function to apply instruction selection and mask annotation.
    """

    def process_func(self, func: Function) -> Function:
        copy_inst_selection = CopyInstructionSelection()
        func = copy_inst_selection(func)

        marker = MaskUsageMarker()
        mask2copy = marker.mark(func)

        mask_annotation = MaskAnnotation(mask2copy)
        func = mask_annotation(func)
        return func


def instruction_selection_pass() -> FunctionPass:
    return InstructionSelectionPass()
