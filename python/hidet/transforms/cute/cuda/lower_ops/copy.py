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
from typing import List, Union, Callable, Optional
from hidet.ir.expr import Expr, Var, Constant, logical_and, cast, deref
from hidet.ir.type import PointerType
from hidet.ir.dtypes import u32, i32, boolean
from hidet.ir.tools import infer_type
from hidet.ir.stmt import DeclareScope
from hidet.lang.cuda import threadIdx
from hidet.ir.primitives import (
    ldg256,
    ldg128,
    ldg64,
    ldg32,
    stg512,
    stg256,
    stg128,
    stg64,
    stg32,
    lds128,
    lds64,
    lds32,
    sts128,
    sts64,
    sts32,
)
from hidet.ir.primitives.cuda.mma import ldmatrix
from hidet.ir.primitives.cuda.cp_async import cp_async

from hidet.ir.cute.ops import Copy, Mask
from hidet.ir.cute.algorithm import TiledCopy
from hidet.ir.cute.layout import (
    TensorLayout,
    composition,
    make_layout,
    max_common_vector,
    left_inverse,
    coalesce,
    filter,
)
from hidet.ir.cute.int_tuple import idx2crd, rank, compact_col_major, flatten, depth, shape_div
from hidet.utils import initialize

from .registry import OpEmitter, Buffer, register_impl

# The bytes of a general purpose register in CUDA is 4
BYTES_PER_GPR = 4
BITS_PER_BYTE = 8


def group(layout: TensorLayout, size: int):
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
        if current_idx * s <= size:
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

    def layout_(shape, stride):
        if len(shape) > 1:
            return TensorLayout(tuple(shape), tuple(stride))
        elif len(shape) == 1:
            return TensorLayout(shape[0], stride[0])
        else:
            return TensorLayout(1)

    return layout_(result_shape, result_stride), layout_(rest_shape, rest_stride)


class CopyInstruction:
    def __init__(
        self,
        apply: Callable,
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

    def match(self, src: Buffer, dst: Buffer, tiled_copy: TiledCopy):
        if src.scope != self.src_scope or dst.scope != self.dst_scope:
            return None
        _, src_tv_layout = tiled_copy.src_tv_layout()
        _, dst_tv_layout = tiled_copy.dst_tv_layout()
        src_thr_layout, src_val_layout = src_tv_layout[0][0], coalesce(
            make_layout(src_tv_layout[0][1], src_tv_layout[1])
        )
        dst_thr_layout, dst_val_layout = dst_tv_layout[0][0], coalesce(
            make_layout(dst_tv_layout[0][1], dst_tv_layout[1])
        )

        src_tv_layout_inst = self._get_layout_in_element(src.var, self.src_layout)
        dst_tv_layout_inst = self._get_layout_in_element(dst.var, self.dst_layout)
        src_thr_layout_inst, src_val_layout_inst = (src_tv_layout_inst[0], src_tv_layout_inst[1])
        dst_thr_layout_inst, dst_val_layout_inst = (dst_tv_layout_inst[0], dst_tv_layout_inst[1])

        src_thr_inst, _ = group(src_thr_layout, filter(src_thr_layout_inst).size())
        src_val_inst, src_val_rest = group(src_val_layout, src_val_layout_inst.size())
        dst_thr_inst, _ = group(dst_thr_layout, dst_thr_layout_inst.size())
        dst_val_inst, dst_val_rest = group(dst_val_layout, dst_val_layout_inst.size())

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

        if cvt_src != cvt_dst:
            return None
        src_val_inst, src_val_rest = group(coalesce(src.layout), src_val_layout_inst.size())
        dst_val_inst, dst_val_rest = group(coalesce(dst.layout), dst_val_layout_inst.size())
        if self.src_scope.is_memory() and src_val_inst.shape != src_val_layout_inst.shape:
            return None
        if self.dst_scope.is_memory() and dst_val_inst.shape != dst_val_layout_inst.shape:
            return None
        return make_layout(src_val_inst, src_val_rest), make_layout(dst_val_inst, dst_val_rest)

    def _get_element_bytes(self, e: Expr):
        ty = infer_type(e)
        assert isinstance(ty, PointerType)
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

    def match(self, src: Buffer, dst: Buffer, tiled_copy: TiledCopy):
        if self.bytes_per_inst > max_common_vector(src.layout, dst.layout) * self._get_element_bytes(src.var):
            return None
        return super().match(src, dst, tiled_copy)

    def __call__(self, src: Expr, dst: Expr, mask: Optional[Expr] = None):
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

    def __call__(self, src: Expr, dst: Expr, mask: Optional[Expr] = None):
        assert mask is None
        assert self.src_scope == DeclareScope.Shared and self.dst_scope == DeclareScope.Register
        reg_ptrs = self._get_register_pointers(dst)
        regs = [deref(cast(ptr, ~u32)) for ptr in reg_ptrs]
        return self.apply(regs, src, shared_space_addr=False, trans=self.trans)


memory_instructions = []


@initialize()
def register_universal_copy_instruction():
    insts = [(256, ldg256), (128, ldg128), (64, ldg64), (32, ldg32)]
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

    insts = [(128, lds128), (64, lds64), (32, lds32)]
    for load_bits, inst in insts:
        shape = (1, load_bits)
        src_layout = TensorLayout(((1), (1, load_bits)), ((1), (1, 1)))
        memory_instructions.append(UniversalCopyInstruction(inst, shape, "shared", "register", src_layout))

    insts = [(512, stg512), (256, stg256), (128, stg128), (64, stg64), (32, stg32)]
    for store_bits, inst in insts:
        shape = (1, store_bits)
        src_layout = TensorLayout(((1), (1, store_bits)), ((1), (1, 1)))
        memory_instructions.append(
            UniversalCopyInstruction(inst, shape, "register", "global", src_layout, require_mask=True)
        )

    insts = [(128, sts128), (64, sts64), (32, sts32)]
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


@register_impl(Mask)
class MaskEmitter(OpEmitter):
    def emit(self, op: Mask, args: List[Expr], output: Buffer):
        from hidet.ir.cute.int_tuple import size

        dst: Var = output.var
        shape, src_thrval_layout = op.tiled_copy.src_tv_layout()
        extents = src_thrval_layout[1].shape
        index = TensorLayout(extents)
        nr_masks = size(extents)
        nr_regs = (nr_masks + u32.nbytes * 8 - 1) // (u32.nbytes * 8)

        extents = extents if isinstance(extents, tuple) else [extents]
        base = Var("base", i32)
        self.declare(base, src_thrval_layout[0][0](threadIdx.x))
        with self.for_grid([nr_regs]) as i:
            self.assign(dst[i], 0)
        with self.for_grid(extents) as indices:
            indices = indices[0] if len(indices) == 1 else indices
            idx = Var("idx", i32)
            mask_idx = Var("mask_idx", i32)
            bit = Var("bit", i32)
            crd = Var("crd", i32)
            pred = Var("pred", boolean)
            self.declare(idx, index(indices))
            self.declare(mask_idx, (idx >> 5))
            self.declare(bit, (idx & 31))
            self.declare(crd, base + src_thrval_layout[1](indices))
            for i, (v, e) in enumerate(zip(args, idx2crd(crd, shape))):
                if i == 0:
                    self.declare(pred, e < v)
                else:
                    self.assign(pred, logical_and(pred, (e < v)))
            self.assign(dst[mask_idx], dst[mask_idx] | (cast(pred, u32) << bit))


@register_impl(Copy)
class CopyEmitter(OpEmitter):
    def instruction_selection(self, op: Copy, args: List[Union[Buffer, Expr]], output: Optional[Buffer]):
        from hidet.ir.cute.int_tuple import size

        candidates = []
        for inst in memory_instructions:
            result = inst.match(args[0], args[1], op.tiled_copy)
            if result is not None:
                candidates.append((inst, *result))
        candidates = sorted(candidates, key=lambda x: -size(x[0].bytes_per_inst))
        return candidates[0]

    def emit(self, op: Copy, args: List[Union[Buffer, Expr]], output: Optional[Buffer]):
        assert isinstance(args[0], Buffer)
        assert isinstance(args[1], Buffer)
        assert len(args) <= 2 or isinstance(args[2], Buffer)
        src: Buffer = args[0]
        dst: Buffer = args[1]
        mask: Optional[Buffer] = args[2] if len(args) >= 3 else None
        src_var = src.var
        src_ty = infer_type(src_var)
        assert isinstance(src_ty, PointerType)
        dst_var = dst.var
        dst_ty = infer_type(dst_var)
        assert isinstance(dst_ty, PointerType)
        shape, _ = op.tiled_copy.src_tv_layout()
        _shape, _ = op.tiled_copy.dst_tv_layout()
        assert shape == _shape
        operands = [src_var, dst_var]
        operand_tys = [src_ty, dst_ty]
        inst, src_layout, dst_layout = self.instruction_selection(op, args, output)

        extents = src_layout[1].shape
        if mask is not None:
            index = TensorLayout(shape)
        extents = extents if isinstance(extents, tuple) else [extents]
        with self.for_grid(extents) as indices:
            indices = indices[0] if len(indices) == 1 else indices
            src_addr, dst_addr = [
                Var(v.hint, PointerType(base_type=t.base_type)) for v, t in zip(operands, operand_tys)
            ]
            self.declare(src_addr, src_var + src_layout[1](indices))
            self.declare(dst_addr, dst_var + dst_layout[1](indices))
            if mask is not None:
                idx = Var("idx", i32)
                mask_idx = Var("mask_idx", i32)
                bit = Var("bit", i32)
                pred = Var("pred", boolean)
                self.declare(idx, index(indices))
                self.declare(mask_idx, (idx >> 5))
                self.declare(bit, (idx & 31))
                self.declare(pred, mask.var[mask_idx] & (Constant(1, u32) << bit))
            else:
                pred = None
            self.append(inst(src_addr, dst_addr, pred))
