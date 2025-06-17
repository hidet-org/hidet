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
# pylint: disable=too-many-lines
from typing import Tuple, List, Dict, Union, Optional, Callable
import functools

from hidet.ir.type import DataType, PointerType, data_type, TensorType, TensorPointerType
from hidet.ir.expr import Add, Address, TensorElement, TensorSlice, Var, Expr, var, deref, cast, if_then_else, Cast
from hidet.ir.stmt import DeclareScope, DeclareStmt, IfStmt
from hidet.ir.tools import TypeInfer, infer_type, simplify, rewrite
from hidet.ir.functors import IRVisitor, IRRewriter
from hidet.ir.dtypes import u8, u32, float16, float16x2

from hidet.ir.cute.expr import Op, CallOp, CConst
from hidet.ir.cute.type import TiledTensorType

from hidet.ir.func import Function
from hidet.transforms.base import FunctionPass
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
from hidet.ir.primitives.cuda.wgmma import WgmmaConfig, wgmma_configs, wgmma_async
from hidet.ir.primitives.cuda.cp_async import cp_async
from hidet.ir.primitives.cuda.atomic import reduce_add, atomic_add
from hidet.ir.primitives.cuda.copy_tma import copy_tensor_g2s, copy_tensor_s2g

from hidet.ir.cute import Swizzle
from hidet.ir.cute.ops import TensorBase, TensorView, Copy, Mask, Mma, PartitionA, PartitionB, Atomic, AtomicAdd
from hidet.ir.cute.algorithm import TiledCopy, TiledMma
from hidet.ir.cute.layout import (
    TiledTensorLayout,
    ComposedTensorLayout,
    TensorLayout,
    LayoutBase,
    composition,
    make_layout,
    max_common_vector,
    left_inverse,
    left_inverse_ignore_zero_strides,
    coalesce,
    filter,
    common_reshape,
    group,
    canonicalize_thread_value_layout,
    prefix_product,
    register_tensor_layout,
    filter_zeros,
)
from hidet.ir.cute.int_tuple import is_tuple, rank, compact_col_major, flatten, depth, idx2crd, product, product_each
from hidet.utils import initialize
from hidet.transforms.cute.analysis import TensorAliasAnalysis
from hidet.logging import logger, setConsoleLevel, stderr_handler, DEBUG
from .resolve_bank_conflict import ApplySharedMemoryLayoutUpdate
from .lower_ops import Buffer, TmaTensor, tma_tensor
from .tma_layout_utils import (
    common_reshape_per_dim,
    coalesce_gmem_shape_and_smem_shape,
    split_shapes,
    coalesce_per_dim,
    get_last_dim_strides,
)


def expr_to_buffer(expr: Expr, layout_: Union[TiledTensorLayout, ComposedTensorLayout, TensorLayout] = None) -> Buffer:
    """
    Convert an expression to a buffer
    Buffer is a wrapper around a variable that represents a tensor and is used to pass
    tensor information during IR lowering.
    """
    expr_ty: TiledTensorType = infer_type(expr)
    assert isinstance(expr_ty, TiledTensorType)
    layout: Union[TiledTensorLayout, ComposedTensorLayout, TensorLayout] = (
        layout_ if layout_ is not None else expr_ty.layout
    )
    dtype: DataType = expr_ty.dtype
    scope: DeclareScope = expr_ty.scope
    if scope.is_register() and isinstance(layout, TiledTensorLayout):
        layout = register_tensor_layout(layout.val_layout())
    return Buffer(buffer=expr, offset=None, dtype=dtype, scope=scope, layout=layout)


def convert_layout_type_to_layout(tensor2layout: Dict[TensorBase, str]):
    _tensor2layout: Dict[TensorBase, LayoutBase] = {}
    for tensor, layout_type in tensor2layout.items():
        layout = tensor.layout
        dtype = tensor.dtype
        logbits = (dtype.nbits - 1).bit_length()
        if layout_type.startswith("SW32"):
            layout = ComposedTensorLayout(layout, 0, Swizzle(1, 7 - logbits, 3))
        elif layout_type.startswith("SW64"):
            layout = ComposedTensorLayout(layout, 0, Swizzle(2, 7 - logbits, 3))
        elif layout_type.startswith("SW128"):
            layout = ComposedTensorLayout(layout, 0, Swizzle(3, 7 - logbits, 3))
        _tensor2layout[tensor] = layout
    return _tensor2layout


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
        src_thr_layout, src_val_layout = canonicalize_thread_value_layout(src_tv_layout)
        dst_thr_layout, dst_val_layout = canonicalize_thread_value_layout(dst_tv_layout)

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


class TmaCopyInstruction(CopyInstruction):
    def __init__(self, apply: Callable, src_scope: DeclareScope, dst_scope: DeclareScope):
        # In TMA instruction, the shapes and layouts are not determined at the compile
        # time, so we use dummy shapes and layouts to initialize the instruction.
        dummy_shape = (1,)
        # The TMA instruction requires the address alignment to be 128-bit. So, we set
        # the dummy layout to be 128-bit.
        dummy_layout = TensorLayout(((1,), (128,)), ((1,), (1,)))
        super().__init__(apply, dummy_shape, src_scope, dst_scope, dummy_layout, dummy_layout)

    def match(self, src: Buffer, dst: Buffer, tiled_copy: TiledCopy = None):
        """
        Check if the copy operation can be transformed to a TMA instruction.

        Parameters:
            src: The source buffer.
            dst: The destination buffer.
            tiled_copy: The tiled copy information.

        Returns:
            The TMA instruction if the copy operation can be transformed to a TMA instruction,
            otherwise None.
        """
        assert tiled_copy is not None
        # Check if the source and destination scopes match the expected scopes for TMA instructions.
        if src.scope != self.src_scope or dst.scope != self.dst_scope:
            return None
        _, src_tv_layout = tiled_copy.src_tv_layout()
        _, dst_tv_layout = tiled_copy.dst_tv_layout()
        src_t, _ = canonicalize_thread_value_layout(src_tv_layout)
        dst_t, _ = canonicalize_thread_value_layout(dst_tv_layout)

        # Sanity check of the thread layouts. All threads must access the same tensor
        # in the TMA instruction
        if src_t != dst_t:
            return None

        # Sanity check of the stride of the thread layout.
        # The stride being 0 indicates that all threads can access the same tensor
        # memory.
        flat_stride = flatten(src_t.stride)
        if is_tuple(flat_stride):
            if len(flat_stride) != 1:
                return None
            flat_stride = flat_stride[0]
        if flat_stride != 0:
            return None

        if src.scope.is_shared():
            gmem_layout = dst.layout
            smem_layout = src.layout
        else:
            gmem_layout = src.layout
            smem_layout = dst.layout

        swizzle = 'NONE'
        if isinstance(smem_layout, ComposedTensorLayout):
            functor = smem_layout.functor
            logbits = (src.dtype.nbits - 1).bit_length()
            assert isinstance(functor, Swizzle)
            if functor == Swizzle(1, 7 - logbits, 3):
                swizzle = '32B'
            elif functor == Swizzle(2, 7 - logbits, 3):
                swizzle = '64B'
            elif functor == Swizzle(3, 7 - logbits, 3):
                swizzle = '128B'

        divisor = 1
        if src.dtype.is_integer_subbyte():
            divisor = src.dtype.storage.nbits // src.dtype.nbits
        MAX_ELEMENTS_PER_DIM = 256 * divisor

        # Step 1. keep the shape of gmem layout and smem layout the same
        gmem_layout, smem_layout = common_reshape_per_dim(gmem_layout, smem_layout)

        f1 = []
        f2 = []

        def index2coords(idx, shape):
            length = len(shape)
            d = product(shape[: length - 1])
            rst = [0] * (length - 1) + [idx // d]
            return rst

        def split(s, shape):
            length = len(shape)
            d = product(shape[: length - 1])
            rst = shape[:-1] + (s // d,)
            return rst

        for g in gmem_layout:
            f1.append(functools.partial(index2coords, shape=g.shape_tuple))
            f2.append(functools.partial(split, shape=g.shape_tuple))

        gmem_shape = flatten(gmem_layout.shape_tuple)
        gmem_stride = flatten(gmem_layout.stride_tuple)
        smem_shape = flatten(smem_layout.shape_tuple)
        smem_stride = flatten(smem_layout.stride_tuple)
        index = range(len(gmem_shape))
        from builtins import filter as filter_

        # Step 2. The smem layout must be contiguous, so we sort strides and shapes based on ascending order
        # of the smem strides.
        sorted_DS = sorted(filter_(lambda x: x[0] > 0, zip(smem_stride, smem_shape, gmem_stride, gmem_shape, index)))
        smem_stride, smem_shape, gmem_stride, gmem_shape, permute = zip(*sorted_DS)
        gmem_shape, gmem_stride, smem_shape, smem_stride, dims = coalesce_gmem_shape_and_smem_shape(
            gmem_shape, smem_shape, gmem_stride, smem_stride
        )
        if gmem_stride[0] != 1 or (smem_shape[0] * src.dtype.nbits // 8) % 16 != 0:
            return None
        # check alignment
        for ds, dg in zip(smem_stride[1:], gmem_stride[1:]):
            smem_bytes = ds * src.dtype.nbits // 8
            gmem_bytes = dg * src.dtype.nbits // 8
            if (ds != 1 and smem_bytes % 16 != 0) or (dg != 1 and gmem_bytes % 16 != 0):
                return None
        # Step 3. Split the dimension if the number of elements in the dimension is larger than 256
        gmem_shape, gmem_stride, smem_shape, smem_stride, split_dims = split_shapes(
            gmem_shape, smem_shape, gmem_stride, smem_stride, MAX_ELEMENTS_PER_DIM
        )
        gmem_layout = TensorLayout(tuple(gmem_shape), tuple(gmem_stride))
        smem_layout = TensorLayout(tuple(smem_shape), tuple(smem_stride))
        box_shape = smem_shape
        # check if smem layout is contiguous
        smem_layout_ = coalesce(smem_layout)
        if is_tuple(smem_layout_.stride):
            if len(smem_layout_.stride) != 1:
                return None
            smem_stride = smem_layout_.stride[0]
        else:
            smem_stride = smem_layout_.stride
        if smem_stride != 1:
            return None
        dim = rank(gmem_layout.shape)
        # tma only supports tensor dimensions less than and equal to 5
        if dim > 5:
            return None

        # coords_transform is a layout function that converts the global coordinates to
        # the coordinates used in the tma instruction.
        # construct the coords_transform
        def coords_transform(*coords, shp2crd, permute, dims, shape, split_dims):
            coords = list(coords)
            rst = []
            for f, crd in zip(shp2crd, coords):
                rst.extend(f(crd))
            rst_ = [rst[i] for i in permute]
            rst = rst_
            rst_ = []
            idx = 0
            cur_dim_beg = 0
            for cur_dims in dims:
                crd = None
                for _ in cur_dims:
                    if crd is None:
                        crd = rst[idx]
                    else:
                        stride = product(shape[cur_dim_beg + 1 : idx])
                        crd = rst[idx] * stride + crd
                    idx = idx + 1
                cur_dim_beg = idx
                rst_.append(crd)
            rst = list(simplify(e) for e in rst_)
            for i, split_shape_list in split_dims.items():
                crds = idx2crd(rst[i], split_shape_list)
                rst[i] = tuple(crds)
            return flatten(tuple(rst))

        tma_coords_transform = functools.partial(
            coords_transform, shp2crd=f1, permute=permute, dims=dims, shape=smem_shape, split_dims=split_dims
        )

        def extents_transform(*shapes, split_shape_funcs, permute, dims, shape, split_dims):
            shapes = list(shapes)
            rst = []
            for f, s in zip(split_shape_funcs, shapes):
                rst.extend(f(s))
            rst_ = [rst[i] for i in permute]
            rst = rst_
            rst_ = []
            cur_dim_beg = 0
            for cur_dims in dims:
                from hidet.utils.py import prod

                s = prod(rst[cur_dim_beg : cur_dim_beg + len(cur_dims)])
                cur_dim_beg = cur_dim_beg + len(cur_dims)
                rst_.append(s)
            rst = list(simplify(e) for e in rst_)
            for i, shape_list in split_dims.items():
                d = product(shape_list[:-1])
                rst[i] = shape_list[:-1] + (rst[i] // d,)
            return flatten(tuple(rst))

        tma_extents_transform = functools.partial(
            extents_transform, split_shape_funcs=f2, permute=permute, dims=dims, shape=smem_shape, split_dims=split_dims
        )

        tma_strides = flatten(gmem_layout.stride_tuple)
        return dim, box_shape, tma_strides, swizzle, tma_extents_transform, tma_coords_transform

    def __call__(self, smem: Expr, tensor_map: Expr, coords: List[Expr], mbarrier: Optional[Expr] = None):
        coords_rank = len(coords)
        if mbarrier is None:
            return self.apply(coords_rank, smem, tensor_map, *coords)
        else:
            return self.apply(coords_rank, smem, tensor_map, mbarrier, *coords)


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
tma_instructions = []


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

    tma_instructions.append(TmaCopyInstruction(copy_tensor_g2s, "global", "shared"))
    tma_instructions.append(TmaCopyInstruction(copy_tensor_s2g, "shared", "global"))
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

        self.elements_per_inst = self.shape_mnk[0] * self.shape_mnk[1]

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

        a_thr_layout, a_val_layout = canonicalize_thread_value_layout(a_tv_layout)
        b_thr_layout, b_val_layout = canonicalize_thread_value_layout(b_tv_layout)
        c_thr_layout, c_val_layout = canonicalize_thread_value_layout(c_tv_layout)

        if a_val_layout.size() < a_val_layout_inst.size():
            return None
        if b_val_layout.size() < b_val_layout_inst.size():
            return None
        if c_val_layout.size() < c_val_layout_inst.size():
            return None

        a_thr_inst, a_thr_rest = group(a_thr_layout, a_thr_layout_inst.size())
        a_val_inst, a_val_rest = group(a_val_layout, a_val_layout_inst.size())
        b_thr_inst, b_thr_rest = group(b_thr_layout, b_thr_layout_inst.size())
        b_val_inst, b_val_rest = group(b_val_layout, b_val_layout_inst.size())
        c_thr_inst, c_thr_rest = group(c_thr_layout, c_thr_layout_inst.size())
        c_val_inst, c_val_rest = group(c_val_layout, c_val_layout_inst.size())

        # check divisibility
        if any(v is None for v in [a_thr_inst, a_val_inst, b_thr_inst, b_val_inst, c_thr_inst, c_val_inst]):
            return None

        # Step 1: Compute the conversion mapping from the instruction to the tile
        cvt_a = coalesce(
            composition(make_layout(a_thr_inst, a_val_inst), left_inverse_ignore_zero_strides(self.a_layout))
        )
        cvt_b = coalesce(
            composition(make_layout(b_thr_inst, b_val_inst), left_inverse_ignore_zero_strides(self.b_layout))
        )
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
            if m_mode is None or n_mode is None:
                return None, None
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

        if any(v is None for v in [m_mode, n_mode, m_mode_, k_mode, n_mode_, k_mode_]):
            return None

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
            if isinstance(layout, ComposedTensorLayout):
                layout = layout.layout
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


class WgmmaAsyncInstruction(MmaInstruction):
    def __init__(
        self,
        wgmma_config: WgmmaConfig,
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
        b_scope: Union[DeclareScope, str] = "shared",
        c_scope: Union[DeclareScope, str] = "register",
        d_scope: Union[DeclareScope, str] = "register",
        trans_a: Union[bool, None] = None,
        trans_b: Union[bool, None] = False,
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
        self.wgmma_config = wgmma_config
        self.swap_AB = swap_AB
        self._trans_a = trans_a
        self._trans_b = trans_b

        def validate(scope, trans, multiplicand):
            if scope.is_register() and trans is not None:
                raise ValueError(
                    f"Unsupported combination of scope({scope}) and transpose({trans}) for multiplicand {multiplicand}"
                )
            if scope.is_shared() and trans is None:
                raise ValueError(f"Transpose of multiplicand {multiplicand} must be specified for shared scope")

        validate(self.a_scope, self.trans_a, "A")
        validate(self.b_scope, self.trans_b, "B")

    @property
    def trans_a(self):
        if self._trans_a is None:
            return None
        return 1 if self._trans_a else 0

    @property
    def trans_b(self):
        if self._trans_b is None:
            return None
        return 1 if self._trans_b else 0

    def get_core_matrix_layouts(self, dtype: DataType, m: int, k: int, trans: bool):
        if not trans:
            bytes_per_row = 16
            core_matrix_row = 8
            core_matrix_col = bytes_per_row // dtype.nbytes
            layout_interleaved = TensorLayout(
                ((core_matrix_row, m // core_matrix_row), (core_matrix_col, k // core_matrix_col)),
                ((core_matrix_col, k * core_matrix_row), (1, core_matrix_row * core_matrix_col)),
            )
            bytes_per_row = 32
            core_matrix_col = bytes_per_row // dtype.nbytes
            layout_SW32 = TensorLayout((m, core_matrix_col), (core_matrix_col, 1))
            bytes_per_row = 64
            core_matrix_col = bytes_per_row // dtype.nbytes
            layout_SW64 = TensorLayout((m, core_matrix_col), (core_matrix_col, 1))
            bytes_per_row = 128
            core_matrix_col = bytes_per_row // dtype.nbytes
            layout_SW128 = TensorLayout((m, core_matrix_col), (core_matrix_col, 1))
            return [layout_interleaved, layout_SW32, layout_SW64, layout_SW128]
        else:
            bytes_per_row = 16
            core_matrix_row = 8
            core_matrix_col = bytes_per_row // dtype.nbytes
            layout_interleaved = TensorLayout(
                ((core_matrix_row, m // core_matrix_row), (core_matrix_col, k // core_matrix_col)),
                ((1, k * core_matrix_row), (core_matrix_row, core_matrix_row * core_matrix_col)),
            )
            bytes_per_row = 32
            core_matrix_col = bytes_per_row // dtype.nbytes
            # FIXME: if m is not a multiple of core_matrix_col
            if m % core_matrix_col != 0:
                layout_SW32 = None
            else:
                layout_SW32 = TensorLayout(
                    ((core_matrix_col, m // core_matrix_col), (k,)), ((1, k * core_matrix_col), (core_matrix_col,))
                )
            bytes_per_row = 64
            core_matrix_col = bytes_per_row // dtype.nbytes
            if m % core_matrix_col != 0:
                layout_SW64 = None
            else:
                layout_SW64 = TensorLayout(
                    ((core_matrix_col, m // core_matrix_col), (k,)), ((1, k * core_matrix_col), (core_matrix_col,))
                )
            bytes_per_row = 128
            core_matrix_col = bytes_per_row // dtype.nbytes
            if m % core_matrix_col != 0:
                layout_SW128 = None
            else:
                layout_SW128 = TensorLayout(
                    ((core_matrix_col, m // core_matrix_col), (k,)), ((1, k * core_matrix_col), (core_matrix_col,))
                )
            return [layout_interleaved, layout_SW32, layout_SW64, layout_SW128]

    def get_core_matrix_layouts_A(self):
        assert self.a_scope.is_shared()
        m, _, k = self.shape_mnk
        return self.get_core_matrix_layouts(self.a_dtype, m, k, self.trans_a)

    def get_core_matrix_layouts_B(self):
        assert self.b_scope.is_shared()
        _, n, k = self.shape_mnk
        return self.get_core_matrix_layouts(self.b_dtype, n, k, self.trans_b)

    def _get_layout_type(self, layout_type: int, trans: bool):
        _layout_type_dict = {
            (0, True): "Interleaved_T",
            (0, False): "Interleaved_N",
            (1, True): "SW32_T",
            (1, False): "SW32_N",
            (2, True): "SW64_T",
            (2, False): "SW64_N",
            (3, True): "SW128_T",
            (3, False): "SW128_N",
        }
        rst = _layout_type_dict.get((layout_type, trans), None)
        if rst is None:
            raise ValueError(f"Invalid combination of layout type (got:{layout_type}) and transpose(got:{trans})")
        return rst

    def layout_match(self, a_tensor_info, b_tensor_info, tiled_mma: TiledMma):
        _, _, inst_k = self.shape_mnk
        layout_type_a = None
        layout_type_b = None
        if a_tensor_info is not None:
            a_shape, tv = tiled_mma.a_tv_layout()
            _, v = canonicalize_thread_value_layout(tv)
            _, value_inst = self.a_layout
            v_inst, _ = group(v, value_inst.size())
            layout_type = None
            for i, core_matrix_layout in enumerate(self.get_core_matrix_layouts_A()):
                if core_matrix_layout is None:
                    continue
                flat_core_matrix_layout = coalesce(core_matrix_layout)
                _, k_mode = core_matrix_layout
                k_mode_inner, k_mode_outer = group(k_mode, inst_k)
                if k_mode_inner.size() > 1:
                    stride_outer = a_shape[0] * k_mode_inner.size()
                    k_mode_outer_shape = flatten(k_mode_outer.shape_tuple)
                    k_mode_outer_stride = prefix_product(k_mode_outer_shape, stride_outer)
                    v_reshape = make_layout(v_inst, TensorLayout(k_mode_outer_shape, k_mode_outer_stride))
                else:
                    v_reshape = v_inst
                core_matrix_layout = coalesce(composition(a_tensor_info.layout, v_reshape))
                if core_matrix_layout == flat_core_matrix_layout:
                    layout_type = i
                    break
            if layout_type is None:
                return None
            layout_type_a = self._get_layout_type(layout_type, self.trans_a)
        if b_tensor_info is not None:
            b_shape, tv = tiled_mma.b_tv_layout()
            _, v = canonicalize_thread_value_layout(tv)
            _, value_inst = self.b_layout
            v_inst, _ = group(v, value_inst.size())
            layout_type = None
            for i, core_matrix_layout in enumerate(self.get_core_matrix_layouts_B()):
                if core_matrix_layout is None:
                    continue
                flat_core_matrix_layout = coalesce(core_matrix_layout)
                _, k_mode = core_matrix_layout
                k_mode_inner, k_mode_outer = group(k_mode, inst_k)
                if k_mode_inner.size() > 1:
                    stride_outer = b_shape[0] * k_mode_inner.size()
                    k_mode_outer_shape = flatten(k_mode_outer.shape_tuple)
                    k_mode_outer_stride = prefix_product(k_mode_outer_shape, stride_outer)
                    v_reshape = make_layout(v_inst, TensorLayout(k_mode_outer_shape, k_mode_outer_stride))
                else:
                    v_reshape = v_inst
                core_matrix_layout_ = coalesce(composition(b_tensor_info.layout, v_reshape))
                if core_matrix_layout_ == flat_core_matrix_layout:
                    layout_type = i
                    break
            if layout_type is None:
                return None
            layout_type_b = self._get_layout_type(layout_type, self.trans_b)
        return layout_type_a, layout_type_b

    def __call__(self, d: Expr, a: Expr, b: Expr, c: Expr):
        if self.swap_AB:
            return self.apply(self.wgmma_config, b, d, a, trans_a=self.trans_b, trans_b=self.trans_a)
        else:
            return self.apply(self.wgmma_config, a, d, b, trans_a=self.trans_a, trans_b=self.trans_b)


mma_instructions = []


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

    # f8 m16n8k32
    shape = (16, 8, 32)
    a = TensorLayout(((4, 8), (4, 2, 2)), ((64, 1), (16, 8, 256)))
    b = TensorLayout(((4, 8), (4, 2)), ((32, 1), (8, 128)))  # N-major
    c = TensorLayout(((4, 8), (2, 2)), ((32, 1), (16, 8)))
    for acc_dtype in ["f32", "f16"]:
        for ab_dtype in ["f8e4m3", "f8e5m2"]:
            mma_instructions.append(
                MmaSyncInstruction(
                    mma_configs[f"m16n8k32_{ab_dtype}_{acc_dtype}"],
                    False,
                    mma_sync,
                    shape,
                    a,
                    b,
                    c,
                    c,
                    ab_dtype,
                    ab_dtype,
                    acc_dtype,
                )
            )

    for n in range(8, 257, 8):
        for trans_b in [True, False]:
            shape = (64, n, 16)
            # a in register
            a = TensorLayout(((4, 8, 4), (2, 2, 2)), ((128, 1, 16), (64, 8, 512)))
            b = TensorLayout(((128,), (n, 16)), ((0,), (1, n)))
            c = TensorLayout(((4, 8, 4), (2, 2, n // 8)), ((128, 1, 16), (64, 8, 512)))
            mma_instructions.append(
                WgmmaAsyncInstruction(
                    wgmma_configs[f"m64n{n}k16_f16_f16_f16"],
                    False,
                    wgmma_async,
                    shape,
                    a,
                    b,
                    c,
                    c,
                    "f16",
                    "f16",
                    "f16",
                    trans_b=trans_b,
                )
            )
            mma_instructions.append(
                WgmmaAsyncInstruction(
                    wgmma_configs[f"m64n{n}k16_f32_f16_f16"],
                    False,
                    wgmma_async,
                    shape,
                    a,
                    b,
                    c,
                    c,
                    "f16",
                    "f16",
                    "f32",
                    trans_b=trans_b,
                )
            )

            # a in shared
            a = TensorLayout(((128,), (64, 16)), ((0,), (1, 64)))
            b = TensorLayout(((128,), (n, 16)), ((0,), (1, n)))
            c = TensorLayout(((4, 8, 4), (2, 2, n // 8)), ((128, 1, 16), (64, 8, 512)))
            mma_instructions.append(
                WgmmaAsyncInstruction(
                    wgmma_configs[f"m64n{n}k16_f16_f16_f16"],
                    False,
                    wgmma_async,
                    shape,
                    a,
                    b,
                    c,
                    c,
                    "f16",
                    "f16",
                    "f16",
                    a_scope="shared",
                    trans_a=False,
                    trans_b=trans_b,
                )
            )
            mma_instructions.append(
                WgmmaAsyncInstruction(
                    wgmma_configs[f"m64n{n}k16_f32_f16_f16"],
                    False,
                    wgmma_async,
                    shape,
                    a,
                    b,
                    c,
                    c,
                    "f16",
                    "f16",
                    "f32",
                    a_scope="shared",
                    trans_a=False,
                    trans_b=trans_b,
                )
            )

            shape = (64, n, 32)
            # a in register
            a = TensorLayout(((128,), (64, 32)), ((0,), (1, 64)))
            b = TensorLayout(((128,), (n, 32)), ((0,), (1, n)))
            c = TensorLayout(((4, 8, 4), (2, 2, n // 8)), ((128, 1, 16), (64, 8, 512)))
            mma_instructions.append(
                WgmmaAsyncInstruction(
                    wgmma_configs[f"m64n{n}k32_f32_f8e4m3_f8e4m3"],
                    False,
                    wgmma_async,
                    shape,
                    a,
                    b,
                    c,
                    c,
                    "f8e4m3",
                    "f8e4m3",
                    "f32",
                    a_scope="shared",
                    trans_a=False,
                    trans_b=trans_b,
                )
            )

        for trans_a in [True, False]:
            shape = (n, 64, 16)
            # b in register
            a = TensorLayout(((128,), (n, 16)), ((0,), (1, n)))
            b = TensorLayout(((4, 8, 4), (2, 2, 2)), ((128, 1, 16), (64, 8, 512)))
            c = TensorLayout(((4, 8, 4), (2, 2, n // 8)), ((2, n, 16 * n), (1, 8 * n, 8)))
            mma_instructions.append(
                WgmmaAsyncInstruction(
                    wgmma_configs[f"m64n{n}k16_f16_f16_f16"],
                    True,
                    wgmma_async,
                    shape,
                    a,
                    b,
                    c,
                    c,
                    "f16",
                    "f16",
                    "f16",
                    "f16",
                    "shared",
                    "register",
                    trans_a=trans_a,
                    trans_b=None,
                )
            )
            mma_instructions.append(
                WgmmaAsyncInstruction(
                    wgmma_configs[f"m64n{n}k16_f32_f16_f16"],
                    True,
                    wgmma_async,
                    shape,
                    a,
                    b,
                    c,
                    c,
                    "f16",
                    "f16",
                    "f32",
                    "f32",
                    "shared",
                    "register",
                    trans_a=trans_a,
                    trans_b=None,
                )
            )

            # b in shared
            a = TensorLayout(((128,), (n, 16)), ((0,), (1, n)))
            b = TensorLayout(((128,), (64, 16)), ((0,), (1, 64)))
            c = TensorLayout(((4, 8, 4), (2, 2, n // 8)), ((2, n, 16 * n), (1, 8 * n, 8)))
            mma_instructions.append(
                WgmmaAsyncInstruction(
                    wgmma_configs[f"m64n{n}k16_f16_f16_f16"],
                    True,
                    wgmma_async,
                    shape,
                    a,
                    b,
                    c,
                    c,
                    "f16",
                    "f16",
                    "f16",
                    "f16",
                    "shared",
                    "shared",
                    trans_a=trans_a,
                )
            )
            mma_instructions.append(
                WgmmaAsyncInstruction(
                    wgmma_configs[f"m64n{n}k16_f32_f16_f16"],
                    True,
                    wgmma_async,
                    shape,
                    a,
                    b,
                    c,
                    c,
                    "f16",
                    "f16",
                    "f32",
                    "f32",
                    "shared",
                    "shared",
                    trans_a=trans_a,
                )
            )


def get_mma_instructions():
    if len(mma_instructions) == 0:
        register_mma_instruction()
    return mma_instructions


class MmaInstructionSelection(IRRewriter):
    def __init__(self, var2tensor):
        super().__init__()
        self.var2tensor = var2tensor
        self.tensor2layout: Dict[TensorBase, str] = {}

    def _pick_candidate(self, op: Mma, candidates: List[Tuple[MmaInstruction, ...]]):
        candidates = sorted(candidates, key=lambda x: -x[0].elements_per_inst)
        for inst, d_rest, a_rest, b_rest, c_rest in candidates:
            if inst.a_scope.is_shared() or inst.b_scope.is_shared():
                a_tensor_info = None
                b_tensor_info = None
                if inst.a_scope.is_shared():
                    a = op.a
                    assert a in self.var2tensor
                    a_tensor_info = self.var2tensor[a]
                if inst.b_scope.is_shared():
                    b = op.b
                    assert b in self.var2tensor
                    b_tensor_info = self.var2tensor[b]
                result = inst.layout_match(a_tensor_info, b_tensor_info, op.tiled_mma)
                if result is None:
                    continue
                layout_type_a, layout_type_b = result
                if a_tensor_info is not None and layout_type_a is not None:
                    self.tensor2layout[a_tensor_info.tensor] = layout_type_a
                if b_tensor_info is not None and layout_type_b is not None:
                    self.tensor2layout[b_tensor_info.tensor] = layout_type_b
                return inst, d_rest, a_rest, b_rest, c_rest
            else:
                return inst, d_rest, a_rest, b_rest, c_rest
        return None

    def visit_Mma(self, e: Mma):
        args = [self.visit(arg) for arg in e.args]
        bufs = [expr_to_buffer(arg) for arg in args]
        candidates = []
        for inst in get_mma_instructions():
            result = inst.match(*bufs, e.tiled_mma)
            if result is not None:
                candidates.append((inst, *result))
        if len(candidates) == 0:
            raise NotImplementedError(f"Cannot find a suitable instruction for {e}")
        cand = self._pick_candidate(e, candidates)
        if cand is None:
            raise NotImplementedError(f"Cannot find a suitable instruction for {e}")
        inst, d_rest, a_rest, b_rest, c_rest = cand
        annotations: Dict[str, CConst] = {}
        annotations["inst"] = inst
        annotations["a_rest"] = a_rest
        annotations["b_rest"] = b_rest
        annotations["c_rest"] = c_rest
        annotations["d_rest"] = d_rest
        return e.reforward(args, annotations_update=annotations)


class AnnotateLayoutRewritter(IRRewriter):
    def __init__(self, var2tensor, tensor2layout: Dict[TensorBase, str]):
        super().__init__()
        self.var2tensor = var2tensor
        self.tensor2layout: Dict[TensorBase, str] = tensor2layout

    def visit_PartitionA(self, e: PartitionA):
        x_tensor_info = self.var2tensor.get(e.x, None)
        if x_tensor_info is not None and x_tensor_info.tensor in self.tensor2layout:
            x_tensor = x_tensor_info.tensor
            x = self.visit(e.x)
            layout_type = self.tensor2layout[x_tensor]
            return e.reforward([x], annotations_update={"layout_type": layout_type})
        return super().visit_PartitionA(e)

    def visit_PartitionB(self, e: PartitionB):
        x_tensor_info = self.var2tensor.get(e.x, None)
        if x_tensor_info is not None and x_tensor_info.tensor in self.tensor2layout:
            x_tensor = x_tensor_info.tensor
            x = self.visit(e.x)
            layout_type = self.tensor2layout[x_tensor]
            return e.reforward([x], annotations_update={"layout_type": layout_type})
        return super().visit_PartitionB(e)


class TmaCopyInstructionSelection(IRRewriter):
    def __init__(self):
        super().__init__()
        self.infer_type = TypeInfer()
        self.var2tensor: Dict[Var, TensorBase] = {}
        self.tensor2tma_tensors: Dict[TensorBase, List[TmaTensor]] = {}
        self.expr2param_idx: Dict[Expr, int] = {}
        self.var2expr: Dict[Var, Expr] = {}
        self.func_params: List[Var] = []

    def visit_Cast(self, e: Cast):
        new_expr = super().visit_Cast(e)
        if e.expr in self.expr2param_idx:
            expr_ty = self.infer_type(e.expr)
            if isinstance(expr_ty, (PointerType, TensorType, TensorPointerType)):
                idx = self.expr2param_idx[e.expr]
                self.expr2param_idx[e] = idx
        return new_expr

    def visit_DeclareStmt(self, stmt: DeclareStmt):
        new_stmt = super().visit_DeclareStmt(stmt)
        if stmt.init in self.expr2param_idx:
            idx = self.expr2param_idx[stmt.init]
            self.expr2param_idx[stmt.var] = idx
            self.var2expr[stmt.var] = rewrite(stmt.init, self.var2expr)
        return new_stmt

    def visit_TensorElement(self, e: TensorElement):
        new_expr = super().visit_TensorElement(e)
        if e.base in self.expr2param_idx:
            idx = self.expr2param_idx[e.base]
            self.expr2param_idx[e] = idx
        return new_expr

    def visit_Address(self, e: Address):
        new_expr = super().visit_Address(e)
        if e.expr in self.expr2param_idx:
            idx = self.expr2param_idx[e.expr]
            self.expr2param_idx[e] = idx
        return new_expr

    def visit_Add(self, e: Add):
        new_expr = super().visit_Add(e)
        idx = None
        if e.a in self.expr2param_idx:
            idx = self.expr2param_idx[e.a]
        if e.b in self.expr2param_idx:
            idx_ = self.expr2param_idx[e.b]
            #            assert idx is None or idx == idx_
            idx = idx_
        if idx is not None:
            self.expr2param_idx[e] = idx
        return new_expr

    def visit_TensorSlice(self, e: TensorSlice):
        new_expr = super().visit_TensorSlice(e)
        if e.base in self.expr2param_idx:
            idx = self.expr2param_idx[e.base]
            self.expr2param_idx[e] = idx
        return new_expr

    def _select_tma_for_copy(self, e: Copy):
        # can be tma tensors
        src = e.src
        dst = e.dst
        src_ty = self.infer_type(src)
        dst_ty = self.infer_type(dst)

        # get global tensor and layout from src and dst
        global_tensor = None
        if src_ty.scope.is_global():
            global_tensor = src
            if not dst_ty.scope.is_shared():
                return None
        elif dst_ty.scope.is_global():
            global_tensor = dst
            if not src_ty.scope.is_shared():
                return None
        # no global tensor involved, cannot use tma
        if global_tensor is None:
            return None
        tensor_info = self.var2tensor[global_tensor]
        tensor = tensor_info.tensor
        assert isinstance(tensor, TensorView)
        global_layout = tensor.layout
        tile_shape = e.tiled_copy.shape

        src, dst = [expr_to_buffer(arg) for arg in [src, dst]]

        # check if the tensor is a tile of a global tensor
        last_dim_strides = get_last_dim_strides(tile_shape, global_layout)
        if last_dim_strides is None:
            return None

        # canonicalize the layout of src and dst
        smem_last_dim_strides = [None] * len(tile_shape)
        if src.scope.is_global():
            gmem_layout = coalesce_per_dim(src.layout, tile_shape, last_dim_strides)
            smem_layout = coalesce_per_dim(dst.layout, tile_shape, smem_last_dim_strides)
            src.layout = gmem_layout
            dst.layout = smem_layout
        else:
            gmem_layout = coalesce_per_dim(dst.layout, tile_shape, last_dim_strides)
            smem_layout = coalesce_per_dim(src.layout, tile_shape, smem_last_dim_strides)
            dst.layout = gmem_layout
            src.layout = smem_layout

        # tma match
        candidate = None
        for inst in tma_instructions:
            result = inst.match(src, dst, e.tiled_copy)
            if result is not None:
                candidate = (inst, *result)
                break
        if candidate is None:
            return None
        return candidate, tensor, global_layout

    def visit_Copy(self, e: Copy):
        rst = self._select_tma_for_copy(e)
        if rst is not None:
            args = [self.visit(arg) for arg in e.args]
            candidate, tensor, global_layout = rst
            base_ptr = rewrite(tensor.x, self.var2expr)
            assert base_ptr in self.expr2param_idx
            param_idx = self.expr2param_idx[base_ptr]
            param = self.func_params[param_idx]
            # unpack result
            inst, dim, box_shape, tma_strides, swizzle, tma_extents_transform, tma_coords_transform = candidate
            extents = product_each(global_layout.shape_tuple)
            tma_extents = tma_extents_transform(*extents)

            # determine dimensions
            def tma_pointer_functor(arg: Expr, base_ptr: Expr = base_ptr, param: Var = param):
                remap = {param: arg}
                return rewrite(base_ptr, remap)

            pointer_functor = functools.partial(tma_pointer_functor, base_ptr=base_ptr, param=param)

            tma = tma_tensor(
                param_idx, dim, box_shape, tma_strides, tma_extents, pointer_functor=pointer_functor, swizzle=swizzle
            )
            if tensor in self.tensor2tma_tensors:
                tma_tensor_idx = None
                for i, t in enumerate(self.tensor2tma_tensors[tensor]):
                    if t == tma:
                        tma_tensor_idx = i
                        break
                if tma_tensor_idx is None:
                    tma_tensor_idx = len(self.tensor2tma_tensors[tensor])
                    self.tensor2tma_tensors[tensor].append(tma)
            else:
                tma_tensor_idx = 0
                self.tensor2tma_tensors[tensor] = [tma]
            annotations = {}
            annotations["inst"] = inst
            annotations["tma_tensor_idx"] = tma_tensor_idx
            annotations["tma_coords_transform"] = tma_coords_transform
            return e.reforward(args, annotations_update=annotations)
        return super().visit_Copy(e)

    def visit_Function(self, func: Function):
        tensor_alias_analysis = TensorAliasAnalysis()
        tensor_alias_analysis.visit(func)
        self.var2tensor.update(tensor_alias_analysis.var2tensor)

        for i, param in enumerate(func.params):
            self.expr2param_idx[param] = i
        self.func_params = func.params
        return super().visit_Function(func)


class CopyInstructionSelection(IRRewriter):
    def __init__(self, tensor2tma_tensors: Dict[TensorBase, List[TmaTensor]]):
        super().__init__()
        self.infer_type = TypeInfer()
        self.tensor2tma_tensors: Dict[TensorBase, List[TmaTensor]] = tensor2tma_tensors

    def visit_TensorView(self, e: TensorView):
        if e in self.tensor2tma_tensors:
            x = self.visit(e.x)
            tma_tensors = self.tensor2tma_tensors[e]
            annotations = {}
            annotations["tma_tensors"] = tma_tensors
            return e.reforward([x], annotations_update=annotations)
        return super().visit_TensorView(e)

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

        val_stride = register_tensor_layout(val_layout).stride_tuple
        val_shape_fz = filter_zeros(val_stride, val_shape)
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
        if e.annotations is not None and "tma_coords_transform" in e.annotations:
            return e

        args = [self.visit(arg) for arg in e.args]

        bufs = [expr_to_buffer(arg) for arg in args[:2]]
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
            if e.mask in self.old2new:
                mask = self.old2new[e.mask]
            else:
                mask = self.visit(e.mask)
        else:
            mask = None
        if e.mbarrier is not None:
            mbarrier = self.visit(e.mbarrier)
        else:
            mbarrier = None
        if src is e.src and dst is e.dst and mask is e.mask and mbarrier is e.mbarrier:
            return e
        else:
            return e.reforward([src, dst, mask, mbarrier])

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
        origin_handler_level = stderr_handler.level
        if DEBUG_VERBOSE:
            logger.setLevel(DEBUG)
            setConsoleLevel(DEBUG)

        # Step 1: Perform tensor alias analysis
        tensor_alias_analysis = TensorAliasAnalysis()
        tensor_alias_analysis(func)
        var2tensor = tensor_alias_analysis.var2tensor

        # Step 2: Perform instruction selection for mma operations
        mma_inst_selection = MmaInstructionSelection(var2tensor)
        func = mma_inst_selection(func)
        tensor2layout = mma_inst_selection.tensor2layout

        # Step 3: Annotate the layout type for PartitionA and PartitionB
        rewriter = AnnotateLayoutRewritter(var2tensor, tensor2layout)
        func = rewriter(func)

        # Step 4: Convert the shared memory layout to Swizzle Layout
        _tensor2layout = convert_layout_type_to_layout(tensor2layout)
        rewriter = ApplySharedMemoryLayoutUpdate(_tensor2layout, {})
        func = rewriter(func)

        # Step 5: Perform tma instruction selection for copy operations
        tma_inst_selection = TmaCopyInstructionSelection()
        func = tma_inst_selection(func)
        tensor2tma_tensors = tma_inst_selection.tensor2tma_tensors

        # Step 6: Perform instruction selection for copy operations
        copy_inst_selection = CopyInstructionSelection(tensor2tma_tensors)
        func = copy_inst_selection(func)

        # Step 7: Annotate the mask for each copy operation
        marker = MaskUsageMarker()
        mask2user = marker.mark(func)
        mask_annotation = MaskAnnotation(mask2user)
        func = mask_annotation(func)

        if DEBUG_VERBOSE:
            logger.setLevel(origin_level)
            setConsoleLevel(origin_handler_level)
        return func


def instruction_selection_pass() -> FunctionPass:
    return InstructionSelectionPass()
