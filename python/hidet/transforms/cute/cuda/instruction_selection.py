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
from typing import Tuple, List, Dict, Union, Optional, Callable

from hidet.ir.type import DataType, PointerType, data_type
from hidet.ir.expr import Var, Expr, var, deref, cast, if_then_else
from hidet.ir.stmt import DeclareScope
from hidet.ir.tools import TypeInfer, infer_type
from hidet.ir.functors import IRVisitor, IRRewriter
from hidet.ir.dtypes import u8, u32, float16, float16x2

from hidet.ir.cute.expr import Op, CallOp, CConst
from hidet.ir.cute.type import TiledTensorType

from hidet.ir.func import Function
from hidet.transforms.base import FunctionPass
from hidet.ir.stmt import DeclareStmt, IfStmt
from hidet.ir.builders import StmtBuilder


from hidet.ir.primitives import (
    ldg128,
    ldg64,
    ldg32,
    ldg16,
    stg128,
    stg64,
    stg32,
    stg16,
    lds128,
    lds64,
    lds32,
    lds16,
    lds8,
    sts128,
    sts64,
    sts32,
    sts16,
    sts8,
)
from hidet.ir.primitives.cuda.mma import ldmatrix, mma_sync, MmaConfig, mma_configs
from hidet.ir.primitives.cuda.cp_async import cp_async
from hidet.ir.primitives.cuda.atomic import reduce_add, atomic_add

from hidet.ir.cute.ops import Copy, Mask, Mma, Atomic, AtomicAdd
from hidet.ir.cute.algorithm import TiledCopy, TiledMma
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
    common_reshape,
    group,
    canonical_thread_value_layout,
)
from hidet.ir.cute.int_tuple import rank, compact_col_major, flatten, depth
from hidet.utils import initialize
from hidet.logging import logger, DEBUG
from .lower_ops import Buffer


# The bits of a general purpose register in CUDA is 32
BITS_PER_GPR = 32
BITS_PER_BYTE = 8

DEBUG_VERBOSE = False


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

        self.alignment = None
        if self.src_scope.is_memory():
            self.alignment = self.src_layout[1].size() // BITS_PER_BYTE

        if self.dst_scope.is_memory():
            if self.alignment is None:
                self.alignment = self.dst_layout[1].size() // BITS_PER_BYTE
            else:
                alignment = self.dst_layout[1].size() // BITS_PER_BYTE
                assert alignment == self.alignment

    def logical_match(self, src: Buffer, dst: Buffer, tiled_copy: TiledCopy):
        if src.scope != self.src_scope or dst.scope != self.dst_scope:
            return None
        src_element_bits = self._get_element_bits(src.buffer)
        dst_element_bits = self._get_element_bits(dst.buffer)
        if (
            src_element_bits > self.bytes_per_inst * BITS_PER_BYTE
            or dst_element_bits > self.bytes_per_inst * BITS_PER_BYTE
        ):
            return None

        src_tv_layout_inst = self.get_layout_in_element(src.buffer, self.src_layout)
        dst_tv_layout_inst = self.get_layout_in_element(dst.buffer, self.dst_layout)
        src_thr_layout_inst, src_val_layout_inst = (src_tv_layout_inst[0], src_tv_layout_inst[1])
        dst_thr_layout_inst, dst_val_layout_inst = (dst_tv_layout_inst[0], dst_tv_layout_inst[1])

        _, src_tv_layout = tiled_copy.src_tv_layout()
        _, dst_tv_layout = tiled_copy.dst_tv_layout()
        src_thr_layout, src_val_layout = canonical_thread_value_layout(src_tv_layout)
        dst_thr_layout, dst_val_layout = canonical_thread_value_layout(dst_tv_layout)

        # split the thread-value layout into two parts, the element of the first part
        # should be the same as the element of the instruction layout. Then, we can
        # compose the first part with the inverse of the instruction layout to get the
        # conversion mapping.
        src_thr_inst, _ = group(src_thr_layout, filter(src_thr_layout_inst).size(), filter_zero=False)
        src_val_inst, _ = group(src_val_layout, src_val_layout_inst.size(), filter_zero=True)
        dst_thr_inst, _ = group(dst_thr_layout, dst_thr_layout_inst.size(), filter_zero=False)
        dst_val_inst, _ = group(dst_val_layout, dst_val_layout_inst.size(), filter_zero=True)

        # handle corner cases when enabling non-power-two tiles
        if any(v is None for v in [src_thr_inst, src_val_inst, dst_thr_inst, dst_val_inst]):
            return None

        # calculate the conversion mapping
        cvt_src = coalesce(
            composition(make_layout(src_thr_inst, filter(src_val_inst)), left_inverse(filter(src_tv_layout_inst)))
        )
        cvt_dst = coalesce(
            composition(make_layout(dst_thr_inst, filter(dst_val_inst)), left_inverse(dst_tv_layout_inst))
        )

        logger.debug(f"src_thr_layout:{src_thr_layout}")
        logger.debug(f"src_val_layout:{src_val_layout}")
        logger.debug(f"dst_thr_layout:{dst_thr_layout}")
        logger.debug(f"dst_val_layout:{dst_val_layout}")
        logger.debug(f"dst_thr_layout_inst:{dst_thr_layout_inst}")
        logger.debug(f"dst_val_layout_inst:{dst_val_layout_inst}")
        logger.debug(f"cvt_src: {cvt_src}, cvt_dst: {cvt_dst}")

        return cvt_src == cvt_dst

    def loop_match(self, src: Buffer, dst: Buffer):
        if src.scope != self.src_scope or dst.scope != self.dst_scope:
            return None
        src_tv_layout_inst = self.get_layout_in_element(src.buffer, self.src_layout)
        dst_tv_layout_inst = self.get_layout_in_element(dst.buffer, self.dst_layout)
        src_val_layout_inst = src_tv_layout_inst[1]
        dst_val_layout_inst = dst_tv_layout_inst[1]

        # we split the src and dst thread-value layout into two parts: the first part
        # corresponds to an instruction, while the second part indicates the loop
        # structure of the copy instruction.
        # for example, the thread-value layout is (thread, (value_inst, value_loop))
        # the generated code is
        # for i in range(value_loop):
        #     copy_inst(src[i], dst[i])
        src_val_inst, src_val_loop = group(src.layout, src_val_layout_inst.size(), filter_zero=True)
        dst_val_inst, dst_val_loop = group(dst.layout, dst_val_layout_inst.size(), filter_zero=True)

        # handle corner cases when enabling non-power-two tiles
        if any(v is None for v in [src_val_inst, dst_val_inst]):
            return None

        logger.debug(f"{self.apply}")
        logger.debug(f"src: {src_val_inst}, dst: {dst_val_inst}")
        if self.src_scope.is_memory() and filter(src_val_inst).shape != src_val_layout_inst.shape:
            return None
        if self.dst_scope.is_memory() and filter(dst_val_inst).shape != dst_val_layout_inst.shape:
            return None
        src_val_loop = filter(src_val_loop, False)
        dst_val_loop = filter(dst_val_loop, False)
        if src_val_loop.size() != dst_val_loop.size():
            return None
        src_val_loop, dst_val_loop = common_reshape(src_val_loop, dst_val_loop)
        return make_layout(src_val_inst, src_val_loop), make_layout(dst_val_inst, dst_val_loop)

    def match(self, src: Buffer, dst: Buffer, tiled_copy: TiledCopy = None):
        raise NotImplementedError()

    def _get_element_bits(self, e: Expr):
        ty = infer_type(e)
        assert isinstance(ty, (TiledTensorType, PointerType))
        if isinstance(ty, TiledTensorType):
            return ty.dtype.nbits
        else:
            return ty.base_type.nbits

    def _get_register_pointers(self, e: Expr):
        element_bits = self._get_element_bits(e)
        assert element_bits <= BITS_PER_GPR
        incr = BITS_PER_GPR // element_bits
        pointers: List[Expr] = []
        for inc in range(0, (self.bytes_per_inst * BITS_PER_BYTE) // element_bits, incr):
            pointers.append(e + inc)
        return pointers

    def get_layout_in_element(self, e: Expr, layout: TensorLayout):
        """
        The thread-value layouts of the copy instructions are in units of bits, while the thread-value layouts
        of a tensor are in units of elements. So, before we can proceed, we should convert the layouts in bits
        to layouts in elements. This function is used to do this.

        Parameters:
            e (Expr): The tensor expression, used to extract the element data type.
            layout (TensorLayout): The thread-value layout in bits.

        Returns:
            layout(TensorLayout): The thread-value layout in elements.
        """
        element_bits = self._get_element_bits(e)
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

    def __call__(self, src: Expr, dst: Expr, mask: Optional[Expr] = None, **kwargs):
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
        logger.debug(f"inst: {self.apply}, src: {src.layout}, dst: {dst.layout}")
        logger.debug(f"{max_common_vector(src.layout, dst.layout)}")
        if self.alignment * BITS_PER_BYTE > max_common_vector(src.layout, dst.layout) * self._get_element_bits(
            src.buffer
        ):
            return None
        if tiled_copy is None or super().logical_match(src, dst, tiled_copy):
            return super().loop_match(src, dst)
        return None

    def __call__(self, src: Expr, dst: Expr, mask: Optional[Expr] = None, **kwargs):
        # fallback case
        if self.apply is None:
            if self.bytes_per_inst == 1:
                access_dtype = u8
            else:
                raise NotImplementedError()
            sb = StmtBuilder()
            pointer = cast(dst, ~access_dtype)
            pointer_type = infer_type(pointer)
            pointer_var = Var("addr", pointer_type)
            sb.declare(pointer_var, pointer)
            if mask is not None:
                assert self.require_mask
                with sb.if_then(mask):
                    sb.buffer_store(pointer_var, [0], deref(cast(src, ~access_dtype)))
                return sb.finish()
            else:
                sb.buffer_store(pointer_var, [0], deref(cast(src, ~access_dtype)))
                return sb.finish()
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

    def __call__(self, src: Expr, dst: Expr, mask: Optional[Expr] = None, **kwargs):
        operands: List[Expr] = []
        assert self.src_scope.is_global() and self.dst_scope.is_shared()
        operands.append(dst)
        operands.append(src)
        operands.append(self.bytes_per_inst)
        cache_level = "global" if self.bytes_per_inst == 16 else "always"
        if "evict" in kwargs:
            evict = kwargs["evict"]
            evict = "evict_normal" if evict is None else evict
        else:
            evict = "evict_normal"
        if mask is not None:
            return self.apply(
                *operands,
                src_size=if_then_else(mask, self.bytes_per_inst, 0),
                cache_level=cache_level,
                # prefetch_bytes=128,
                evict_policy=evict,
            )
        else:
            return self.apply(*operands, src_size=None, cache_level=cache_level, evict_policy=evict)


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
        logger.debug(f"inst: {self.apply}, src: {src.layout}, dst: {dst.layout}")
        if tiled_copy is None:
            return None
        elif super().logical_match(src, dst, tiled_copy):
            return super().loop_match(src, dst)
        else:
            return None

    def __call__(self, src: Expr, dst: Expr, mask: Optional[Expr] = None, **kwargs):
        assert mask is None
        assert self.src_scope == DeclareScope.Shared and self.dst_scope == DeclareScope.Register

        reg_ptrs = self._get_register_pointers(dst)
        regs = [deref(cast(ptr, ~u32)) for ptr in reg_ptrs]
        return self.apply(regs, src, shared_space_addr=False, trans=self.trans)


class ReduceInstruction(CopyInstruction):
    def __init__(
        self,
        apply: Callable,
        dtype: Union[DataType, str],
        shape: Tuple[int, ...],
        src_scope: DeclareScope,
        dst_scope: DeclareScope,
        src_layout: TensorLayout,
        dst_layout: TensorLayout = None,
    ):
        dst_layout = src_layout if dst_layout is None else dst_layout
        super().__init__(apply, shape, src_scope, dst_scope, src_layout, dst_layout)
        self.dtype = data_type(dtype)

    def match(self, src: Buffer, dst: Buffer, tiled_copy: TiledCopy = None):
        src_dtype = self._get_element_dtype(src.buffer)
        dst_dtype = self._get_element_dtype(dst.buffer)
        if src_dtype != self.dtype or dst_dtype != self.dtype:
            return None
        if self.alignment * BITS_PER_BYTE > max_common_vector(src.layout, dst.layout) * self._get_element_bits(
            src.buffer
        ):
            return None
        if tiled_copy is None or super().logical_match(src, dst, tiled_copy):
            return super().loop_match(src, dst)
        return None

    def _get_element_dtype(self, e: Expr):
        ty = infer_type(e)
        assert isinstance(ty, (TiledTensorType, PointerType))
        if isinstance(ty, TiledTensorType):
            return ty.dtype
        else:
            return ty.base_type

    def __call__(self, src: Expr, dst: Expr, mask: Optional[Expr] = None):
        if self.apply is atomic_add:
            if mask is not None:
                return IfStmt(mask, self.apply(dst, src[0]))
            else:
                return self.apply(dst, src[0])
        # TODO: red instruction with vector arguments only supports sm_90 or
        # higher. So we manually convert to float16x2 to leverage red.f16x2
        # instruction
        # note: alignment equals the alignment in bytes for the instruction
        vector_type = float16x2 if self.dtype == float16 and self.alignment == 4 else self.dtype
        src_values = self._get_register_pointers(src)
        dst_value = dst
        if self.dtype != vector_type:
            src_values = [deref(cast(val, ~vector_type)) for val in src_values]
            dst_value = cast(dst, ~vector_type)
        else:
            src_values = [deref(val) for val in src_values]
        if mask is not None:
            return IfStmt(mask, self.apply(vector_type, dst_value, src_values))
        else:
            return self.apply(vector_type, dst_value, src_values)


atomic_instructions = []


@initialize()
def register_reduce_instruction():
    insts = [(32, reduce_add), (16, reduce_add)]
    # The following instructions only supports sm_90 or higher, so we currently
    # comment them.
    # insts += [(128, reduce_add), (64, reduce_add)]
    for dtype in ["float16", "float32"]:
        for bits, inst in insts:
            shape = (1, bits)
            src_layout = TensorLayout(((1), (1, bits)), ((1), (1, 1)))
            atomic_instructions.append(
                ReduceInstruction(inst, dtype, shape, "register", "shared", src_layout, src_layout)
            )
            atomic_instructions.append(
                ReduceInstruction(inst, dtype, shape, "register", "global", src_layout, src_layout)
            )

    for dtype in ["float16", "float32"]:
        bits = data_type(dtype).nbits
        shape = (1, bits)
        src_layout = TensorLayout(((1), (1, bits)), ((1), (1, 1)))
        atomic_instructions.append(
            ReduceInstruction(atomic_add, dtype, shape, "register", "shared", src_layout, src_layout)
        )
        atomic_instructions.append(
            ReduceInstruction(atomic_add, dtype, shape, "register", "global", src_layout, src_layout)
        )


memory_instructions = []


@initialize()
def register_universal_copy_instruction():
    global memory_instructions
    # remove ldg256 because instructions wider than 16-bytes do not boost
    # performnce
    # insts = [(256, ldg256), (128, ldg128), (64, ldg64), (32, ldg32), (16, None)]
    insts = [(128, ldg128), (64, ldg64), (32, ldg32), (16, ldg16), (8, None)]
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

    insts = [(128, lds128), (64, lds64), (32, lds32), (16, lds16), (8, lds8)]
    for load_bits, inst in insts:
        shape = (1, load_bits)
        src_layout = TensorLayout(((1), (1, load_bits)), ((1), (1, 1)))
        memory_instructions.append(UniversalCopyInstruction(inst, shape, "shared", "register", src_layout))

    # remove stg512, stg256 because instructions wider than 16-bytes do not boost
    # performnce
    # insts = [(512, stg512), (256, stg256), (128, stg128), (64, stg64), (32, stg32), (16, None)]
    insts = [(128, stg128), (64, stg64), (32, stg32), (16, stg16), (8, None)]
    for store_bits, inst in insts:
        shape = (1, store_bits)
        src_layout = TensorLayout(((1), (1, store_bits)), ((1), (1, 1)))
        memory_instructions.append(
            UniversalCopyInstruction(inst, shape, "register", "global", src_layout, require_mask=True)
        )

    insts = [(128, sts128), (64, sts64), (32, sts32), (16, sts16), (8, sts8)]
    for store_bits, inst in insts:
        shape = (1, store_bits)
        src_layout = TensorLayout(((1), (1, store_bits)), ((1), (1, 1)))
        memory_instructions.append(UniversalCopyInstruction(inst, shape, "register", "shared", src_layout))
    memory_instructions = sorted(memory_instructions, key=lambda x: -x.bytes_per_inst)


@initialize()
def register_ldmatrix():
    global memory_instructions
    for nr_mat in [1, 2, 4]:
        for trans in [True, False]:
            shape = (8 * nr_mat, 128)
            src_layout = TensorLayout(((8 * nr_mat, 32 // (8 * nr_mat)), (128)), ((1, 0), (8 * nr_mat)))
            if trans:
                dst_layout = TensorLayout(((4, 8), (16, 2, nr_mat)), ((2, 128 * nr_mat), (8 * nr_mat, 1, 8)))
            else:
                dst_layout = TensorLayout(((4, 8), (32, nr_mat)), ((32 * nr_mat, 1), (8 * nr_mat, 8)))
            memory_instructions.append(LdMatrix(ldmatrix, shape, "shared", "register", src_layout, dst_layout, trans))
    memory_instructions = sorted(memory_instructions, key=lambda x: -x.bytes_per_inst)


class MmaInstruction:
    def __init__(
        self,
        apply: Union[Callable, None],
        shape_mnk,
        a_layout: TensorLayout,
        b_layout: TensorLayout,
        c_layout: TensorLayout,
        d_layout: TensorLayout,
        a_dtype: Union[DataType, str],
        b_dtype: Union[DataType, str],
        c_dtype: Union[DataType, str],
        d_dtype: Union[DataType, str] = None,
        a_scope: Union[DeclareScope, str] = "register",
        b_scope: Union[DeclareScope, str] = "register",
        c_scope: Union[DeclareScope, str] = "register",
        d_scope: Union[DeclareScope, str] = "register",
    ):
        self.apply: Callable = apply
        self.shape_mnk = shape_mnk
        # thread-value layout in k-major
        self.a_layout = a_layout
        # thread-value layout in k-major
        self.b_layout = b_layout
        self.c_layout = c_layout
        self.d_layout = d_layout
        d_dtype = c_dtype if d_dtype is None else d_dtype

        self.a_dtype = data_type(a_dtype)
        self.b_dtype = data_type(b_dtype)
        self.c_dtype = data_type(c_dtype)
        self.d_dtype = data_type(c_dtype)

        def _scope(scope: Union[DeclareScope, str]):
            return scope if isinstance(scope, DeclareScope) else DeclareScope.from_str(scope)

        self.a_scope = _scope(a_scope)
        self.b_scope = _scope(b_scope)
        self.c_scope = _scope(c_scope)
        self.d_scope = _scope(d_scope)

    def match(self, d: Buffer, a: Buffer, b: Buffer, c: Buffer, tiled_mma: TiledMma):
        got_scope = [a.scope, b.scope, c.scope, d.scope]
        expected_scope = [self.a_scope, self.b_scope, self.c_scope, self.d_scope]
        if any(got != exp for got, exp in zip(got_scope, expected_scope)):
            return None
        got_dtype = [self._get_element_dtype(buf.buffer) for buf in [a, b, c, d]]
        expected_dtype = [self.a_dtype, self.b_dtype, self.c_dtype, self.d_dtype]
        if any(got != exp for got, exp in zip(got_dtype, expected_dtype)):
            return None

        inst_m, inst_n, _ = self.shape_mnk

        shape_mk, a_tv_layout = tiled_mma.a_tv_layout()
        shape_nk, b_tv_layout = tiled_mma.b_tv_layout()
        _, c_tv_layout = tiled_mma.c_tv_layout()
        m, _ = shape_mk
        n, _ = shape_nk

        a_thr_layout_inst, a_val_layout_inst = (self.a_layout[0], self.a_layout[1])
        b_thr_layout_inst, b_val_layout_inst = (self.b_layout[0], self.b_layout[1])
        c_thr_layout_inst, c_val_layout_inst = (self.c_layout[0], self.c_layout[1])

        a_thr_layout, a_val_layout = canonical_thread_value_layout(a_tv_layout)
        b_thr_layout, b_val_layout = canonical_thread_value_layout(b_tv_layout)
        c_thr_layout, c_val_layout = canonical_thread_value_layout(c_tv_layout)

        a_thr_inst, a_thr_rest = group(a_thr_layout, a_thr_layout_inst.size())
        a_val_inst, a_val_rest = group(a_val_layout, a_val_layout_inst.size())
        b_thr_inst, b_thr_rest = group(b_thr_layout, b_thr_layout_inst.size())
        b_val_inst, b_val_rest = group(b_val_layout, b_val_layout_inst.size())
        c_thr_inst, c_thr_rest = group(c_thr_layout, c_thr_layout_inst.size())
        c_val_inst, c_val_rest = group(c_val_layout, c_val_layout_inst.size())

        # Step 1: Compute the conversion mapping from the instruction to the tile
        cvt_a = coalesce(composition(make_layout(a_thr_inst, a_val_inst), left_inverse(self.a_layout)))
        cvt_b = coalesce(composition(make_layout(b_thr_inst, b_val_inst), left_inverse(self.b_layout)))
        cvt_c = coalesce(composition(make_layout(c_thr_inst, c_val_inst), left_inverse(self.c_layout)))

        def split_mn(cvt: TensorLayout, m: int, inst_m: int):
            """
            Split the tensor layout into m-mode and n-mode based on the given dimensions.

            Args:
                cvt (TensorLayout): The conversion tensor layout.
                m (int): The m dimension size.
                inst_m (int): The instruction m dimension size.

            Returns:
                Tuple[TensorLayout, TensorLayout]: The m-mode and n-mode tensor layouts.
            """
            from builtins import filter as filter_

            flat_shape = flatten(cvt.shape_tuple)
            flat_stride = flatten(cvt.stride_tuple)
            flat_cvt = TensorLayout(flat_shape, flat_stride)
            m_mode, n_mode = group(flat_cvt, inst_m)
            flat_shape = m_mode.shape_tuple + n_mode.shape_tuple
            flat_stride = m_mode.stride_tuple + n_mode.stride_tuple
            mode_m = list(filter_(lambda t: t[1] < m, zip(flat_shape, flat_stride)))
            mode_n = list(filter_(lambda t: t[1] >= m, zip(flat_shape, flat_stride)))
            m_shape = tuple(s for s, _ in mode_m)
            m_stride = tuple(d for _, d in mode_m)
            n_shape = tuple(s for s, _ in mode_n)
            n_stride = tuple(d // m for _, d in mode_n)
            m_mode = coalesce(TensorLayout(m_shape, m_stride))
            n_mode = coalesce(TensorLayout(n_shape, n_stride))
            return m_mode, n_mode

        # Step 2: split the m_mode, n_mode, and k_mode from the conversion mapping
        m_mode, n_mode = split_mn(cvt_c, m, inst_m)
        m_mode_, k_mode = split_mn(cvt_a, m, inst_m)
        n_mode_, k_mode_ = split_mn(cvt_b, n, inst_n)

        # Step 3: check if the m_mode, n_mode, and k_mode are consistent
        # If not, the mma instruction should not be selected
        if m_mode != m_mode_ or n_mode != n_mode_ or k_mode != k_mode_:
            return None

        def group_val(arg: Buffer, inst: TensorLayout):
            """
            Group the tensor layout based on the instruction size.
            This function is used to group the value layout into two parts
            based on the instruction size. The first part is the instruction
            mode and the second part is the rest mode.

            Args:
                arg (Buffer): The buffer argument.
                inst (TensorLayout): The instruction tensor layout.

            Returns:
                Tuple[TensorLayout, TensorLayout]: The grouped tensor layouts.
            """
            layout = arg.layout
            if isinstance(layout, TensorLayout):
                return group(layout, inst.size())
            else:
                assert isinstance(layout, TiledTensorLayout)
                assert arg.scope.is_register()
                val = layout.val_layout()
                val = TensorLayout(val.shape)
                return group(val, inst.size())

        # Step 4: extract the rest layout from the thread-value layout for each tensor
        _, a_val_rest_prime = group_val(a, a_val_layout_inst)
        _, b_val_rest_prime = group_val(b, b_val_layout_inst)
        _, c_val_rest_prime = group_val(c, c_val_layout_inst)

        def split_rest(rest: TensorLayout, rest_prime: TensorLayout, m: int):
            """
            Split the rest tensor layout into m-mode and n-mode based on the given dimensions.

            Args:
                rest (TensorLayout): The rest tensor layout.
                rest_prime (TensorLayout): The rest prime tensor layout.
                m (int): The m dimension size.

            Returns:
                Tuple[TensorLayout, TensorLayout, TensorLayout, TensorLayout]: The m-mode,
                n-mode, m-prime, and n-prime tensor layouts.
            """

            from builtins import filter as filter_

            rest, rest_prime = common_reshape(rest, rest_prime)
            flat_shape = flatten(rest.shape_tuple)
            flat_stride = flatten(rest.stride_tuple)
            prime_shape = flatten(rest_prime.shape_tuple)
            prime_stride = flatten(rest_prime.stride_tuple)
            mode_m = list(filter_(lambda t: t[1] < m, zip(flat_shape, flat_stride, prime_shape, prime_stride)))
            mode_n = list(filter_(lambda t: t[1] >= m, zip(flat_shape, flat_stride, prime_shape, prime_stride)))
            m_shape, m_stride, m_prime_shape, m_prime_stride = list(zip(*mode_m)) if len(mode_m) > 0 else [1] * 4
            n_shape, n_stride, n_prime_shape, n_prime_stride = list(zip(*mode_n)) if len(mode_n) > 0 else [1] * 4
            n_stride = tuple(d // m for d in n_stride) if isinstance(n_stride, tuple) else n_stride
            m_mode = coalesce(TensorLayout(m_shape, m_stride))
            n_mode = coalesce(TensorLayout(n_shape, n_stride))
            m_prime = coalesce(TensorLayout(m_prime_shape, m_prime_stride))
            n_prime = coalesce(TensorLayout(n_prime_shape, n_prime_stride))
            return m_mode, n_mode, m_prime, n_prime

        # Step 5: extract the m_mode, n_mode, m_prime, and n_prime from the rest layout
        m_val_rest, n_val_rest, cm, cn = split_rest(c_val_rest, c_val_rest_prime, m)
        m_val_rest_, k_val_rest, am, ak = split_rest(a_val_rest, a_val_rest_prime, m)
        n_val_rest_, k_val_rest_, bn, bk = split_rest(b_val_rest, b_val_rest_prime, n)

        # Step 6: check if the m_mode, n_mode, m_prime, and n_prime are consistent
        # If not, the mma instruction should not be selected
        if m_val_rest != m_val_rest_ or n_val_rest != n_val_rest_ or k_val_rest != k_val_rest_:
            return None

        def constraint_abc(c: TensorLayout, a: TensorLayout, b: TensorLayout, m: int):
            """
            Check if the rest layouts of A, B, and C belonging to the thread mode are consistent.

            Args:
                c (TensorLayout): The C tensor layout.
                a (TensorLayout): The A tensor layout.
                b (TensorLayout): The B tensor layout.
                m (int): The m dimension size.

            Returns:
                bool: True if the layouts are consistent, False otherwise.
            """
            flat_shape = flatten(c.shape_tuple)
            flat_stride = flatten(c.stride_tuple)
            a_stride = tuple(d if d < m else 0 for d in flat_stride)
            b_stride = tuple(d // m if d >= m else 0 for d in flat_stride)
            a_ = coalesce(TensorLayout(flat_shape, a_stride))
            b_ = coalesce(TensorLayout(flat_shape, b_stride))
            return coalesce(a) == a_ and coalesce(b) == b_

        # Step 7: check if the rest layouts of A, B, and C belonging to the thread mode are consistent
        if not constraint_abc(c_thr_rest, a_thr_rest, b_thr_rest, m):
            return None

        cm, am = common_reshape(cm, am)
        cn, bn = common_reshape(cn, bn)
        ak, bk = common_reshape(ak, bk)

        # Step 8: reorganize the rest layout of A, B, and C
        # Note: the rest layout could be used to generate the loop of gemm operation
        # The loop looks like:
        # for i in grid(cm):
        #     for j in grid(cn):
        #         c[i, j] = 0
        #         for k in grid(ak):
        #             c[i, j] += a[i, k] * b[k, j]
        # unlike the scalar gemm operation, the m,n, and k indices of the loop are multi-dimensional
        # coordinates other than scalars.
        return make_layout(cm, cn), make_layout(am, ak), make_layout(bn, bk), make_layout(cm, cn)

    def _get_element_dtype(self, e: Expr):
        ty = infer_type(e)
        assert isinstance(ty, (TiledTensorType, PointerType))
        if isinstance(ty, TiledTensorType):
            return ty.dtype
        else:
            return ty.base_type

    def _get_element_bits(self, e: Expr):
        return self._get_element_dtype(e).nbits

    def _get_register_pointers(self, e: Expr):
        element_bits = self._get_element_bits(e)
        assert element_bits <= BITS_PER_GPR
        incr = BITS_PER_GPR // element_bits
        pointers: List[Expr] = []
        for inc in range(0, (self.bytes_per_inst * BITS_PER_BYTE) // element_bits, incr):
            pointers.append(e + inc)
        return pointers

    def __call__(self, d: Expr, a: Expr, b: Expr, c: Expr):
        raise NotImplementedError()


class FmaSyncInstruction(MmaInstruction):
    def __init__(
        self,
        apply: Callable,
        shape_mnk,
        a_layout: TensorLayout,
        b_layout: TensorLayout,
        c_layout: TensorLayout,
        d_layout: TensorLayout,
        a_dtype: Union[DataType, str],
        b_dtype: Union[DataType, str],
        c_dtype: Union[DataType, str],
        d_dtype: Union[DataType, str] = None,
    ):
        super().__init__(
            apply,
            shape_mnk,
            a_layout,
            b_layout,
            c_layout,
            d_layout,
            a_dtype,
            b_dtype,
            c_dtype,
            d_dtype,
            "register",
            "register",
            "register",
            "register",
        )

    def __call__(self, d: Expr, a: Expr, b: Expr, c: Expr):
        return


class MmaSyncInstruction(MmaInstruction):
    def __init__(
        self,
        mma_config: MmaConfig,
        swap_AB: bool,
        apply: Callable,
        shape_mnk,
        a_layout: TensorLayout,
        b_layout: TensorLayout,
        c_layout: TensorLayout,
        d_layout: TensorLayout,
        a_dtype: Union[DataType, str],
        b_dtype: Union[DataType, str],
        c_dtype: Union[DataType, str],
        d_dtype: Union[DataType, str] = None,
        a_scope: Union[DeclareScope, str] = "register",
        b_scope: Union[DeclareScope, str] = "register",
        c_scope: Union[DeclareScope, str] = "register",
        d_scope: Union[DeclareScope, str] = "register",
    ):
        super().__init__(
            apply,
            shape_mnk,
            a_layout,
            b_layout,
            c_layout,
            d_layout,
            a_dtype,
            b_dtype,
            c_dtype,
            d_dtype,
            a_scope,
            b_scope,
            c_scope,
            d_scope,
        )
        self.mma_config = mma_config
        self.swap_AB = swap_AB

    def __call__(self, d: Expr, a: Expr, b: Expr, c: Expr):
        if self.swap_AB:
            return self.apply(self.mma_config, b, a, d)
        else:
            return self.apply(self.mma_config, a, b, d)


mma_instructions = []


@initialize()
def register_mma_instruction():
    shape = (16, 8, 16)
    a = TensorLayout(((4, 8), (2, 2, 2)), ((32, 1), (16, 8, 128)))
    b = TensorLayout(((4, 8), (2, 2)), ((16, 1), (8, 64)))
    c = TensorLayout(((4, 8), (2, 2)), ((32, 1), (16, 8)))
    mma_instructions.append(
        MmaSyncInstruction(mma_configs["m16n8k16_f16_f16"], False, mma_sync, shape, a, b, c, c, "f16", "f16", "f16")
    )
    # float32 accumulation
    mma_instructions.append(
        MmaSyncInstruction(mma_configs["m16n8k16_f16_f32"], False, mma_sync, shape, a, b, c, c, "f16", "f16", "f32")
    )

    shape = (8, 16, 16)
    a = TensorLayout(((4, 8), (2, 2)), ((16, 1), (8, 64)))
    b = TensorLayout(((4, 8), (2, 2, 2)), ((32, 1), (16, 8, 128)))
    c = TensorLayout(((4, 8), (2, 2)), ((2, 8), (1, 64)))
    mma_instructions.append(
        MmaSyncInstruction(mma_configs["m16n8k16_f16_f16"], True, mma_sync, shape, a, b, c, c, "f16", "f16", "f16")
    )
    # float32 accumulation
    mma_instructions.append(
        MmaSyncInstruction(mma_configs["m16n8k16_f16_f32"], True, mma_sync, shape, a, b, c, c, "f16", "f16", "f32")
    )

    shape = (16, 8, 16)
    a = TensorLayout(((4, 8), (2, 2, 2)), ((32, 1), (16, 8, 128)))
    b = TensorLayout(((4, 8), (2, 2)), ((16, 1), (8, 64)))
    c = TensorLayout(((4, 8), (2, 2)), ((32, 1), (16, 8)))
    mma_instructions.append(
        MmaSyncInstruction(mma_configs["m16n8k16_bf16_f32"], False, mma_sync, shape, a, b, c, c, "bf16", "bf16", "f32")
    )

    shape = (8, 16, 16)
    a = TensorLayout(((4, 8), (2, 2)), ((16, 1), (8, 64)))
    b = TensorLayout(((4, 8), (2, 2, 2)), ((32, 1), (16, 8, 128)))
    c = TensorLayout(((4, 8), (2, 2)), ((2, 8), (1, 64)))
    mma_instructions.append(
        MmaSyncInstruction(mma_configs["m16n8k16_bf16_f32"], True, mma_sync, shape, a, b, c, c, "bf16", "bf16", "f32")
    )


class MmaInstructionSelection(IRRewriter):
    def __init__(self):
        super().__init__()

    def visit_Mma(self, e: Mma):
        args = [self.visit(arg) for arg in e.args]

        def arg_to_buffer(arg: Expr):
            arg_ty: TiledTensorType = infer_type(arg)
            layout: Union[TiledTensorLayout, ComposedTensorLayout, TensorLayout] = arg_ty.layout
            dtype: DataType = arg_ty.dtype
            scope: DeclareScope = arg_ty.scope
            return Buffer(buffer=arg, offset=None, dtype=dtype, scope=scope, layout=layout)

        bufs = [arg_to_buffer(arg) for arg in args]
        candidates = []
        for inst in mma_instructions:
            result = inst.match(*bufs, e.tiled_mma)
            if result is not None:
                candidates.append((inst, *result))
        if len(candidates) == 0:
            raise NotImplementedError(f"Cannot find a suitable instruction for {e}")
        # candidates = sorted(candidates, key=lambda x: -x[0].bytes_per_inst)
        inst, d_rest, a_rest, b_rest, c_rest = candidates[0]
        annotations: Dict[str, CConst] = {}
        annotations["inst"] = inst
        annotations["a_rest"] = a_rest
        annotations["b_rest"] = b_rest
        annotations["c_rest"] = c_rest
        annotations["d_rest"] = d_rest
        return e.reforward(args, annotations_update=annotations)


class CopyInstructionSelection(IRRewriter):
    def __init__(self):
        super().__init__()
        self.infer_type = TypeInfer()

    def visit_Atomic(self, e: Atomic):
        if isinstance(e, AtomicAdd):
            return self.visit_AtomicAdd(e)
        return super().visit_Atomic(e)

    def visit_AtomicAdd(self, e: AtomicAdd):
        args = [self.visit(arg) for arg in e.args]

        src = args[0]
        src_ty = self.infer_type(src)
        src_layout: TiledTensorLayout = src_ty.layout
        tiled_copy: TiledCopy = TiledCopy.from_tiled_tensor_layout(src_layout)
        val_layout: TensorLayout = src_layout.val_layout()
        val_shape = val_layout.shape_tuple
        val_stride = val_layout.stride_tuple
        from hidet.ir.cute.ops.partition import reg_tensor_stride
        from hidet.ir.cute import prefix_product, filter_zeros

        val_shape_fz = filter_zeros(val_stride, val_shape)
        val_stride = reg_tensor_stride(prefix_product(val_shape_fz), val_stride)
        src_layout: TensorLayout = TensorLayout(val_shape_fz, val_stride)
        src_dtype: DataType = src_ty.dtype
        src_scope: DeclareScope = src_ty.scope
        bufs = [Buffer(buffer=src, offset=None, dtype=src_dtype, scope=src_scope, layout=src_layout)]

        dst = args[1]
        dst_ty = self.infer_type(dst)
        dst_layout: TensorLayout = dst_ty.layout
        dst_layout: TensorLayout = composition(dst_layout, val_layout)
        dst_dtype: DataType = dst_ty.dtype
        dst_scope: DeclareScope = dst_ty.scope
        bufs += [Buffer(buffer=dst, offset=None, dtype=dst_dtype, scope=dst_scope, layout=dst_layout)]

        candidates = []
        for inst in atomic_instructions:
            result = inst.match(bufs[0], bufs[1], tiled_copy)
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

    def visit_Copy(self, e: Copy):
        args = [self.visit(arg) for arg in e.args]

        def arg_to_buffer(arg: Expr):
            arg_ty: TiledTensorType = self.infer_type(arg)
            layout: Union[TiledTensorLayout, ComposedTensorLayout, TensorLayout] = arg_ty.layout
            dtype: DataType = arg_ty.dtype
            scope: DeclareScope = arg_ty.scope
            return Buffer(buffer=arg, offset=None, dtype=dtype, scope=scope, layout=layout)

        bufs = [arg_to_buffer(arg) for arg in args[:2]]
        candidate = None
        logger.debug("============================================================")
        logger.debug(f"copy: {e}")
        for i, buf in enumerate(bufs):
            logger.debug(f"operand {i}: {buf.scope}")
        for inst in memory_instructions:
            result = inst.match(bufs[0], bufs[1], e.tiled_copy)
            if result is not None:
                candidate = (inst, *result)
                break
        if candidate is None:
            raise NotImplementedError(f"Cannot find a suitable instruction for {e}")
        inst, src_layout, dst_layout = candidate
        logger.debug(f"inst: {inst.apply}")
        logger.debug("============================================================")
        annotations: Dict[str, CConst] = {}
        annotations["inst"] = inst
        annotations["src_layout"] = src_layout
        annotations["dst_layout"] = dst_layout
        return e.reforward(args, annotations_update=annotations)


class MaskUsageMarker(IRVisitor):
    def __init__(self):
        super().__init__()
        self.mask2user: Dict[Mask, Copy] = {}
        self.var2mask: Dict[Var, Mask] = {}
        self.var2user: Dict[Var, Op] = {}

    def mark(self, func: Function):
        self.visit(func)
        return self.mask2user

    def visit_DeclareStmt(self, stmt: DeclareStmt):
        v = stmt.var
        if isinstance(stmt.init, CallOp):
            call = stmt.init
            op = call.op
            if isinstance(op, Mask):
                self.var2mask[v] = op
                if v in self.var2user:
                    user = self.var2user[v]
                    self.mask2user[op] = user

    def visit_Atomic(self, e: Atomic):
        if e.mask is not None:
            if isinstance(e.mask, Var):
                v = e.mask
                self.var2user[v] = e
                if v in self.var2mask:
                    mask = self.var2mask[v]
                    self.mask2user[mask] = e
            else:
                assert isinstance(e.mask, CallOp)
                mask = e.mask.op
                self.mask2user[mask] = e

    def visit_Copy(self, e: Copy):
        if e.mask is not None:
            if isinstance(e.mask, Var):
                v = e.mask
                self.var2user[v] = e
                if v in self.var2mask:
                    mask = self.var2mask[v]
                    self.mask2user[mask] = e
            else:
                assert isinstance(e.mask, CallOp)
                mask = e.mask.op
                self.mask2user[mask] = e


class MaskAnnotation(IRRewriter):
    def __init__(self, mask2user: Dict[Mask, Copy]):
        super().__init__()
        self.mask2user = mask2user
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

    def visit_Atomic(self, e: Atomic):
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
        if e not in self.mask2user:
            raise ValueError("Operator mask{e} hasn't been marked, there should be some bugs.")
        user = self.mask2user[e]
        if isinstance(user, Copy):
            src_layout = user.annotations["src_layout"]
            inst_len = src_layout[0].count()
            tiled_copy: TiledCopy = e.tiled_copy
            _, src_thrval_layout = tiled_copy.src_tv_layout()
            val_layout = coalesce(make_layout(src_thrval_layout[0][1], src_thrval_layout[1]))
            _, rest_layout = group(val_layout, inst_len, filter_zero=True)
            # We need to filter the 0-stride dimension because we won't issue the instruction
            # for the index in 0-stride dimension. This makes the mask align with the copy
            # operation.
            rest_layout = filter(rest_layout, False)
            annotations: Dict[str, CConst] = {}
            annotations["rest_layout"] = rest_layout
            return e.reforward(args, annotations_update=annotations)
        else:
            raise NotImplementedError()


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
        origin_level = logger.level
        if DEBUG_VERBOSE:
            logger.setLevel(DEBUG)

        mma_inst_selection = MmaInstructionSelection()
        func = mma_inst_selection(func)

        copy_inst_selection = CopyInstructionSelection()
        func = copy_inst_selection(func)

        marker = MaskUsageMarker()
        mask2user = marker.mark(func)

        mask_annotation = MaskAnnotation(mask2user)
        func = mask_annotation(func)

        if DEBUG_VERBOSE:
            logger.setLevel(origin_level)
        return func


def instruction_selection_pass() -> FunctionPass:
    return InstructionSelectionPass()
