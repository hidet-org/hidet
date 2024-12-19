import hidet
from typing import List, Dict

from hidet.lang import attrs
from hidet.lang import register_tensor
from hidet.lang.cuda import (
    blockIdx,
    threadIdx,
    dynamic_shared_memory,
    syncthreads,
    cp_async_wait_all,
    cp_async,
    cp_async_commit_group,
    cp_async_wait_group,
)
from hidet.ir.primitives.cuda.mutex import release_seq_semaphore, acquire_seq_semaphore
from hidet.ir.primitives.cuda.atomic import atomic_add
from hidet.ir.expr import symbol_var
from hidet.ir.stmt import asm

from hidet.ir.cute.layout import TensorLayout, make_layout
from hidet.ir.cute.layout import Level
from hidet.ir.cute.algorithm import MmaAtom, TiledMma, auto_copy
from hidet.ir.cute.ops import (
    make_tensor,
    tensor_view,
    partition_src,
    partition_dst,
    mask,
    copy,
    mma,
    sub_tensor,
    rearrange,
    arithmetic,
    cast,
    fill,
)

from hidet.ir.cute import auto_layout, layout_auto
from hidet.ir.cute import composition, coalesce, logical_divide

from hidet.utils.py import cdiv
from hidet.utils import initialize
from hidet.lang import grid

from hidet.lang.types import i32, f16, u4
from hidet.ir.type import DataType
from hidet.runtime.compiled_module import CompiledFunction

import torch


def wrapper_func(func, *tensors: torch.Tensor):
    s1 = hidet.cuda.current_stream().handle()
    s2 = torch.cuda.current_stream().cuda_stream
    if s1 == s2:
        func(*tensors)
    else:
        s = hidet.cuda.ExternalStream(s2)
        with hidet.cuda.stream(s):
            func(*tensors)


def depreprocess_weight(weight: torch.Tensor):
    m, n = weight.shape
    dtype = weight.dtype
    element_size = weight.element_size()
    pack_factor = element_size * 8 // u4.nbits
    n = n * pack_factor
    w = torch.empty(m, n // pack_factor, dtype=dtype, device='cuda')
    bm, bn = 64, 64
    threads = 128
    assert m % bm == 0 and n % bn == 0

    weight = weight.cuda()
    basic_block = TensorLayout(((8, 2), (2, 4, 2)), ((4, 2), (1, 64, 32)))
    m_mode, n_mode = basic_block
    n_shape = n_mode.shape + (m // n_mode.size(),)
    n_stride = n_mode.stride + (basic_block.cosize(),)
    n_mode = TensorLayout(n_shape, n_stride)
    m_shape = m_mode.shape + (n // m_mode.size(),)
    cosize = m // 16 * basic_block.cosize()
    m_stride = m_mode.stride + (cosize,)
    m_mode = TensorLayout(m_shape, m_stride)
    gmem_layout = make_layout(n_mode, m_mode)

    layout = TensorLayout((m, n))
    tile = TensorLayout((bm, bn), (1, m))
    tile = logical_divide(layout, tile)
    tile = composition(gmem_layout, tile)
    gmem, strides = tile
    m_stride, n_stride = strides.stride
    m_stride //= bm
    n_stride //= bn * m

    with hidet.script_module() as script_module:

        @hidet.script
        def func(wi: u4[n, m], wo: u4[m, n]):
            attrs.func_kind = "cuda_kernel"
            attrs.cuda.block_dim = threads
            attrs.cuda.grid_dim = cdiv(m, bm) * cdiv(n, bn), 1
            attrs.cuda.dynamic_smem_bytes = 0

            pid = blockIdx.x
            num_pid_n = cdiv(n, bn)
            pid_m = pid // num_pid_n
            pid_n = pid % num_pid_n

            tg_wi = tensor_view(wi[pid_n * bn * n_stride :, pid_m * bm * m_stride :], gmem, "global")
            tr_wi = make_tensor(u4, auto_layout, "register")

            txgx_wi = partition_src(tg_wi, auto_copy())
            txrx_wi = partition_dst(tr_wi, auto_copy())
            copy(auto_copy((bm, bn)), txgx_wi, txrx_wi)

            tr_w = cast(tr_wi, f16)
            tr_w_cvt = rearrange(tr_w, auto_layout, "register")
            tr_wo = cast(tr_w_cvt, u4)

            tg_wo = tensor_view(wo[pid_m * bm :, pid_n * bn :], TensorLayout((bm, bn), (n, 1)), "global")
            txgx_wo = partition_dst(tg_wo, auto_copy())
            txrx_wo = partition_src(tr_wo, auto_copy())

            copy(auto_copy((bm, bn)), txrx_wo, txgx_wo)

    func = script_module.build()
    wrapper_func(func, weight, w)
    return w


def cast_u4_to_f16_interleaved(t: torch.Tensor):
    m, n = t.shape
    n = n * t.element_size() * 8 // u4.nbits
    ts = torch.empty(m, n, dtype=torch.float16, device='cuda')
    bm, bn = 64, 64
    threads = 128

    from hidet.ir.primitives.cuda.cvt import cast_u4x8_to_f16x8_interleaved_func

    cast_interleaved = cast_u4x8_to_f16x8_interleaved_func()

    t = t.cuda()
    with hidet.script_module() as script_module:

        @hidet.script
        def func(wi: u4[m, n], wo: f16[m, n]):
            attrs.func_kind = "cuda_kernel"
            attrs.cuda.block_dim = threads
            attrs.cuda.grid_dim = cdiv(m, bm) * cdiv(n, bn), 1
            attrs.cuda.dynamic_smem_bytes = 0

            pid = blockIdx.x
            num_pid_n = cdiv(n, bn)
            pid_m = pid // num_pid_n
            pid_n = pid % num_pid_n

            mski = mask(auto_copy(), [m - pid_m * bm, n - pid_n * bn])
            tg_wi = tensor_view(wi[pid_m * bm :, pid_n * bn :], TensorLayout((bm, bn), (n, 1)), "global")
            tr_wi = make_tensor(u4, auto_layout, "register")

            txgx_wi = partition_src(tg_wi, auto_copy())
            txrx_wi = partition_dst(tr_wi, auto_copy())
            copy(auto_copy((bm, bn)), txgx_wi, txrx_wi, mski)

            tr_w = arithmetic(tr_wi, op=cast_interleaved)
            tr_wo = rearrange(tr_w, auto_layout, "register")

            msko = mask(auto_copy(), [m - pid_m * bm, n - pid_n * bn])
            tg_wo = tensor_view(wo[pid_m * bm :, pid_n * bn :], TensorLayout((bm, bn), (n, 1)), "global")
            txgx_wo = partition_dst(tg_wo, auto_copy())
            txrx_wo = partition_src(tr_wo, auto_copy())

            copy(auto_copy((bm, bn)), txrx_wo, txgx_wo, msko)

    func = script_module.build()
    wrapper_func(func, t, ts)
    return ts


def cast_u4_to_f16(t: torch.Tensor):
    m, n = t.shape
    n = n * t.element_size() * 8 // u4.nbits
    ts = torch.empty(m, n, dtype=torch.float16, device='cuda')
    bm, bn = 64, 64
    threads = 128

    t = t.cuda()
    with hidet.script_module() as script_module:

        @hidet.script
        def func(wi: u4[m, n], wo: f16[m, n]):
            attrs.func_kind = "cuda_kernel"
            attrs.cuda.block_dim = threads
            attrs.cuda.grid_dim = cdiv(m, bm) * cdiv(n, bn), 1
            attrs.cuda.dynamic_smem_bytes = 0

            pid = blockIdx.x
            num_pid_n = cdiv(n, bn)
            pid_m = pid // num_pid_n
            pid_n = pid % num_pid_n

            mski = mask(auto_copy(), [m - pid_m * bm, n - pid_n * bn])
            tg_wi = tensor_view(wi[pid_m * bm :, pid_n * bn :], TensorLayout((bm, bn), (n, 1)), "global")
            tr_wi = make_tensor(u4, auto_layout, "register")

            txgx_wi = partition_src(tg_wi, auto_copy())
            txrx_wi = partition_dst(tr_wi, auto_copy())
            copy(auto_copy((bm, bn)), txgx_wi, txrx_wi, mski)

            tr_w = cast(tr_wi, f16)
            tr_wo = rearrange(tr_w, auto_layout, "register")

            msko = mask(auto_copy(), [m - pid_m * bm, n - pid_n * bn])
            tg_wo = tensor_view(wo[pid_m * bm :, pid_n * bn :], TensorLayout((bm, bn), (n, 1)), "global")
            txgx_wo = partition_dst(tg_wo, auto_copy())
            txrx_wo = partition_src(tr_wo, auto_copy())

            copy(auto_copy((bm, bn)), txrx_wo, txgx_wo, msko)

    func = script_module.build()
    wrapper_func(func, t, ts)
    return ts


def cast_f16_to_u4(t: torch.Tensor):
    m, n = t.shape
    ts = torch.empty(m, n // 8, dtype=torch.int32, device='cuda')
    bm, bn = 64, 64
    threads = 128

    with hidet.script_module() as script_module:

        @hidet.script
        def func(wi: f16[m, n], wo: u4[m, n]):
            attrs.func_kind = "cuda_kernel"
            attrs.cuda.block_dim = threads
            attrs.cuda.grid_dim = cdiv(m, bm) * cdiv(n, bn), 1
            attrs.cuda.dynamic_smem_bytes = 0

            pid = blockIdx.x
            num_pid_n = cdiv(n, bn)
            pid_m = pid // num_pid_n
            pid_n = pid % num_pid_n

            mski = mask(auto_copy(), [m - pid_m * bm, n - pid_n * bn])
            tg_wi = tensor_view(wi[pid_m * bm :, pid_n * bn :], TensorLayout((bm, bn), (n, 1)), "global")
            tr_wi = make_tensor(f16, auto_layout, "register")

            txgx_wi = partition_src(tg_wi, auto_copy())
            txrx_wi = partition_dst(tr_wi, auto_copy())
            copy(auto_copy((bm, bn)), txgx_wi, txrx_wi, mski)

            tr_w = cast(tr_wi, u4)
            tr_wo = rearrange(tr_w, auto_layout, "register")

            msko = mask(auto_copy(), [m - pid_m * bm, n - pid_n * bn])
            tg_wo = tensor_view(wo[pid_m * bm :, pid_n * bn :], TensorLayout((bm, bn), (n, 1)), "global")
            txgx_wo = partition_dst(tg_wo, auto_copy())
            txrx_wo = partition_src(tr_wo, auto_copy())

            copy(auto_copy((bm, bn)), txrx_wo, txgx_wo, msko)

    func = script_module.build()
    wrapper_func(func, t, ts)
    return ts


def preprocess_weight(weight: torch.Tensor):
    m, n = weight.shape
    dtype = weight.dtype
    element_size = weight.element_size()
    pack_factor = element_size * 8 // u4.nbits
    n = n * pack_factor
    w = torch.empty(m, n // pack_factor, dtype=dtype, device='cuda')
    bm, bn = 64, 64
    threads = 128
    assert m % bm == 0 and n % bn == 0

    if not weight.is_contiguous():
        weight = weight.contiguous()
    basic_block = TensorLayout(((8, 2), (2, 4, 2)), ((4, 2), (1, 64, 32)))
    m_mode, n_mode = basic_block
    n_shape = n_mode.shape + (m // n_mode.size(),)
    n_stride = n_mode.stride + (basic_block.cosize(),)
    n_mode = TensorLayout(n_shape, n_stride)
    m_shape = m_mode.shape + (n // m_mode.size(),)
    cosize = m // 16 * basic_block.cosize()
    m_stride = m_mode.stride + (cosize,)
    m_mode = TensorLayout(m_shape, m_stride)
    gmem_layout = make_layout(n_mode, m_mode)

    layout = TensorLayout((m, n))
    tile = TensorLayout((bm, bn), (1, m))
    tile = logical_divide(layout, tile)
    tile = composition(gmem_layout, tile)
    gmem, strides = tile
    m_stride, n_stride = strides.stride
    m_stride //= bm
    n_stride //= bn * m

    from hidet.ir.primitives.cuda.cvt import cast_u4x8_to_f16x8_interleaved_func

    cast_interleaved = cast_u4x8_to_f16x8_interleaved_func()

    with hidet.script_module() as script_module:

        @hidet.script
        def func(wi: u4[m, n], wo: u4[n, m]):
            attrs.func_kind = "cuda_kernel"
            attrs.cuda.block_dim = threads
            attrs.cuda.grid_dim = cdiv(m, bm) * cdiv(n, bn), 1
            attrs.cuda.dynamic_smem_bytes = 0

            pid = blockIdx.x
            num_pid_n = cdiv(n, bn)
            pid_m = pid // num_pid_n
            pid_n = pid % num_pid_n

            tg_wi = tensor_view(wi[pid_m * bm :, pid_n * bn :], TensorLayout((bm, bn), (n, 1)), "global")
            tr_wi = make_tensor(u4, auto_layout, "register")

            txgx_wi = partition_src(tg_wi, auto_copy())
            txrx_wi = partition_dst(tr_wi, auto_copy())
            copy(auto_copy((bm, bn)), txgx_wi, txrx_wi)

            tr_w = arithmetic(tr_wi, op=cast_interleaved)
            tr_w_cvt = rearrange(tr_w, auto_layout, "register")
            tr_wo = cast(tr_w_cvt, u4)

            tg_wo = tensor_view(wo[pid_n * bn * n_stride :, pid_m * bm * m_stride :], gmem, "global")
            txgx_wo = partition_dst(tg_wo, auto_copy())
            txrx_wo = partition_src(tr_wo, auto_copy())

            copy(auto_copy((bm, bn)), txrx_wo, txgx_wo)

    func = script_module.build()
    wrapper_func(func, weight, w)
    return w


_predefined_tiled_mma: List[TiledMma] = []


@initialize()
def register_tiled_mma():
    a = TensorLayout(((4, 8), (2, 2)), ((16, 1), (8, 64)))
    b = TensorLayout(((4, 8), (2, 2, 2)), ((32, 1), (16, 8, 128)))
    c = TensorLayout(((4, 8), (2, 2)), ((2, 8), (1, 64)))
    mma_atom = MmaAtom("warp", (8, 16, 16), a, b, c, c, (1, 2))
    warp_in_threadblock = Level("warp", "thread_block", (1, 2), TensorLayout((1, 2)), (1, 1))
    tiled_mma = TiledMma(mma_atom, [warp_in_threadblock])
    _predefined_tiled_mma.append(tiled_mma)

    a = TensorLayout(((4, 8), (2, 2)), ((16, 1), (8, 64)))
    b = TensorLayout(((4, 8), (2, 2, 2)), ((32, 1), (16, 8, 128)))
    c = TensorLayout(((4, 8), (2, 2)), ((2, 8), (1, 64)))
    mma_atom = MmaAtom("warp", (8, 16, 16), a, b, c, c, (1, 2))
    warp_in_threadblock = Level("warp", "thread_block", (1, 4), TensorLayout((1, 4)), (1, 1))
    tiled_mma = TiledMma(mma_atom, [warp_in_threadblock])
    _predefined_tiled_mma.append(tiled_mma)

    a = TensorLayout(((4, 8), (2, 2)), ((16, 1), (8, 64)))
    b = TensorLayout(((4, 8), (2, 2, 2)), ((32, 1), (16, 8, 128)))
    c = TensorLayout(((4, 8), (2, 2)), ((2, 8), (1, 64)))
    mma_atom = MmaAtom("warp", (8, 16, 16), a, b, c, c, (2, 1))
    warp_in_threadblock = Level("warp", "thread_block", (1, 4), TensorLayout((1, 4)), (1, 1))
    tiled_mma = TiledMma(mma_atom, [warp_in_threadblock])
    _predefined_tiled_mma.append(tiled_mma)

    a = TensorLayout(((4, 8), (2, 2)), ((16, 1), (8, 64)))
    b = TensorLayout(((4, 8), (2, 2, 2)), ((32, 1), (16, 8, 128)))
    c = TensorLayout(((4, 8), (2, 2)), ((2, 8), (1, 64)))
    mma_atom = MmaAtom("warp", (8, 16, 16), a, b, c, c, (2, 1))
    warp_in_threadblock = Level("warp", "thread_block", (1, 8), TensorLayout((1, 8)), (1, 1))
    tiled_mma = TiledMma(mma_atom, [warp_in_threadblock])
    _predefined_tiled_mma.append(tiled_mma)

    a = TensorLayout(((4, 8), (2, 2)), ((16, 1), (8, 64)))
    b = TensorLayout(((4, 8), (2, 2, 2)), ((32, 1), (16, 8, 128)))
    c = TensorLayout(((4, 8), (2, 2)), ((2, 8), (1, 64)))
    mma_atom = MmaAtom("warp", (8, 16, 16), a, b, c, c, (4, 1))
    warp_in_threadblock = Level("warp", "thread_block", (1, 8), TensorLayout((1, 8)), (1, 1))
    tiled_mma = TiledMma(mma_atom, [warp_in_threadblock])
    _predefined_tiled_mma.append(tiled_mma)

    a = TensorLayout(((4, 8), (2, 2)), ((16, 1), (8, 64)))
    b = TensorLayout(((4, 8), (2, 2, 2)), ((32, 1), (16, 8, 128)))
    c = TensorLayout(((4, 8), (2, 2)), ((2, 8), (1, 64)))
    mma_atom = MmaAtom("warp", (8, 16, 16), a, b, c, c, (4, 2))
    warp_in_threadblock = Level("warp", "thread_block", (4, 2), TensorLayout((4, 2)), (1, 2))
    tiled_mma = TiledMma(mma_atom, [warp_in_threadblock])
    _predefined_tiled_mma.append(tiled_mma)


def canonicalize(layout: TensorLayout):
    return layout[0][0], coalesce(make_layout(layout[0][1], layout[1]))


class Config:
    def __init__(self, tiled_mma: TiledMma, block_k: int, parallel_k_slices: int = 1):
        self.tiled_mma = tiled_mma
        self.block_k = block_k
        a_shape, _ = self.tiled_mma.a_tv_layout()
        b_shape, _ = self.tiled_mma.b_tv_layout()

        block_m, k_tile = a_shape
        block_n, _ = b_shape
        self.block_m = block_m
        self.block_n = block_n
        self.parallel_k_slices = parallel_k_slices
        self.k_tile = k_tile

    def __str__(self):
        indent = " " * 2
        return (
            "{\n"
            + f"{indent}block_m: {self.block_m},"
            + f"{indent}block_n: {self.block_n},"
            + f"{indent}block_k: {self.block_k},"
            + f"{indent}parallel_k_parts: {self.parallel_k_parts},"
            + "}"
        )

    @property
    def threads(self):
        _, c_tv_layout = self.tiled_mma.c_tv_layout()
        c_t, _ = canonicalize(c_tv_layout)
        return c_t.size()

    @property
    def a_elements(self):
        _, a_tv_layout = self.tiled_mma.a_tv_layout()
        _, a_v = canonicalize(a_tv_layout)
        return a_v.size()

    @property
    def b_elements(self):
        _, b_tv_layout = self.tiled_mma.b_tv_layout()
        _, b_v = canonicalize(b_tv_layout)
        return b_v.size()

    @property
    def c_elements(self):
        _, c_tv_layout = self.tiled_mma.c_tv_layout()
        _, c_v = canonicalize(c_tv_layout)
        return c_v.size()

    def scale_elements(self, group_size: int = 64):
        _, bn, bk = self.thread_block_shape
        _, b_tv_layout = self.tiled_mma.b_tv_layout()
        _, b_v = canonicalize(b_tv_layout)
        if bk > group_size:
            scale_v = composition(TensorLayout((bn, (group_size, bk // group_size)), (1, (0, bn * group_size))), b_v)
        else:
            scale_v = composition(TensorLayout((bn, bk), (1, 0)), b_v)
        return scale_v.count()

    def bias_elements(self, group_size):
        return self.scale_elements(group_size)

    @property
    def parallel_k_parts(self):
        return self.parallel_k_slices

    @property
    def thread_block_shape(self):
        return self.block_m, self.block_n, self.block_k

    def dynamic_smem_bytes_per_stage(self, a_dtype: DataType, b_dtype: DataType = u4, group_size: int = 64):
        smem_a = self.block_m * self.block_k * a_dtype.nbytes
        smem_b = self.block_n * self.block_k * b_dtype.nbits // 8
        smem_scale = self.block_n * cdiv(self.block_k, group_size) * f16.nbytes
        smem_bias = self.block_n * cdiv(self.block_k, group_size) * f16.nbytes
        return smem_a + smem_b + smem_scale + smem_bias

    def stages(self, a_dtype: DataType, b_dtype: DataType = u4, group_size: int = 64):
        dyn_smem_bytes_per_stage = self.dynamic_smem_bytes_per_stage(a_dtype, b_dtype, group_size)
        # the magic number is tuned for RTX4090
        MAXIMUM_DYNAMIC_SMEM_SIZE = 49152
        MAXIMUM_STAGES = 7
        stages = MAXIMUM_DYNAMIC_SMEM_SIZE // dyn_smem_bytes_per_stage
        stages = min(stages, MAXIMUM_STAGES)
        return stages

    def dynamic_smem_bytes(self, a_dtype: DataType, b_dtype: DataType = u4, group_size: int = 64, stages=None):
        dyn_smem_bytes_per_stage = self.dynamic_smem_bytes_per_stage(a_dtype, b_dtype, group_size)
        if stages is None:
            stages = self.stages(a_dtype, b_dtype, group_size)
        return dyn_smem_bytes_per_stage * stages


class FpAIntBGemm:
    def __init__(self, name: str, k: int, n: int, group_size: int, weight_dtype: DataType = u4):
        self.name = name
        self.k = k
        self.n = n
        self.group_size = group_size
        self.weight_dtype = weight_dtype
        self.functions: Dict[str, CompiledFunction] = {}
        self.m_symbol_name = f"m_{self.name}"
        self._compile()

    def __call__(self, a: torch.Tensor, b: torch.Tensor, scale: torch.Tensor, zeros: torch.Tensor):
        from hidet.ffi import runtime_api

        m, _ = a.shape
        _, n = b.shape
        n = n * b.element_size() * 8 // self.weight_dtype.nbits
        bm, bn, _, parallel_k_parts, func = self._heuristic_dispatch(m)
        runtime_api.set_symbol_value(self.m_symbol_name, m)
        c = torch.empty((m, n), dtype=torch.float16, device="cuda")
        c_parallel_k_parts = torch.empty((parallel_k_parts, m, n), dtype=torch.float16, device="cuda")
        grid_m, grid_n = cdiv(m, bm), cdiv(n, bn)
        lock = torch.empty((grid_m, grid_n), dtype=torch.int32, device="cuda")
        wrapper_func(func, a, b, c, scale, zeros, c_parallel_k_parts, lock)
        return c

    def _heuristic_dispatch(self, m: int):
        k = self.k
        n = self.n
        if k >= 2 * n:
            if m <= 8:
                bm, bn, bk = 8, 64, 256
                parallel_k_parts = 8
                buffer = "single_buffer"
            elif m <= 16:
                bm, bn, bk = 16, 64, 256
                parallel_k_parts = 8
                buffer = "single_buffer"
            elif m <= 32:
                bm, bn, bk = 32, 128, 128
                parallel_k_parts = 4
                buffer = "multi_buffer"
            elif m <= 64:
                bm, bn, bk = 32, 128, 128
                parallel_k_parts = 4
                buffer = "multi_buffer"
            else:
                bm, bn, bk = 32, 128, 128
                parallel_k_parts = 1
                buffer = "multi_buffer"
        else:
            if m <= 8:
                bm, bn, bk = 8, 128, 128
                parallel_k_parts = 4
                buffer = "multi_buffer"
            elif m <= 16:
                bm, bn, bk = 16, 128, 128
                parallel_k_parts = 4
                buffer = "multi_buffer"
            elif m <= 32:
                bm, bn, bk = 32, 128, 128
                parallel_k_parts = 1
                buffer = "multi_buffer"
            elif m <= 64:
                bm, bn, bk = 32, 128, 128
                parallel_k_parts = 1
                buffer = "multi_buffer"
            else:
                bm, bn, bk = 32, 128, 128
                parallel_k_parts = 1
                buffer = "multi_buffer"
        indent = " " * 2
        tile_config = (
            "{\n"
            + f"{indent}block_m: {bm},"
            + f"{indent}block_n: {bn},"
            + f"{indent}block_k: {bk},"
            + f"{indent}parallel_k_parts: {parallel_k_parts},"
            + "}"
        )
        key = f"{buffer}\n{tile_config}"
        assert key in self.functions
        func = self.functions[key]
        return bm, bn, bk, parallel_k_parts, func

    def _deduce_gmem_layout(self, block_k: int, block_n: int, stages: int = 1):
        # magic gmem and smem layout that coalesce global memory load and
        # resolve shared memory bank conflict
        basic_block = TensorLayout(((8, 2), (2, 4, 2)), ((4, 2), (1, 64, 32)))
        m_mode, n_mode = basic_block
        n_shape = n_mode.shape + (block_k // n_mode.size(),)
        n_stride = n_mode.stride + (basic_block.cosize(),)
        n_mode_ = TensorLayout(n_shape, n_stride)
        m_shape = m_mode.shape + (block_n // m_mode.size(),)
        cosize = block_k // 16 * basic_block.cosize()
        m_stride = m_mode.stride + (cosize,)
        m_mode_ = TensorLayout(m_shape, m_stride)
        if stages > 1:
            smem_layout = make_layout(m_mode_, n_mode_)
            stage_layout = TensorLayout(stages, smem_layout.cosize())
            smem_layout = make_layout(m_mode_, n_mode_, stage_layout)
        else:
            smem_layout = make_layout(m_mode_, n_mode_)

        n_shape = n_mode.shape + (self.k // n_mode.size(),)
        n_stride = n_mode.stride + (basic_block.cosize(),)
        n_mode_ = TensorLayout(n_shape, n_stride)
        m_shape = m_mode.shape + (block_n // m_mode.size(),)
        cosize = self.k // 16 * basic_block.cosize()
        m_stride = m_mode.stride + (cosize,)
        m_mode_ = TensorLayout(m_shape, m_stride)
        gmem_layout = make_layout(m_mode_, n_mode_)
        return gmem_layout, smem_layout

    def _compile(self):
        single_buffer_configs = []
        multi_buffer_configs = []

        n = self.n
        k = self.k
        for tiled_mma in _predefined_tiled_mma:
            c_shape, _ = tiled_mma.c_tv_layout()
            bm, bn = c_shape
            if bn == 128:
                bk = 128
            elif bn == 64:
                bk = 256
            if bm >= 128:
                bk = 64
            if k >= 2 * n:
                parallel_k_parts = 8
                single_buffer_configs.append(Config(tiled_mma, bk))
                single_buffer_configs.append(Config(tiled_mma, bk, parallel_k_parts))
            else:
                parallel_k_parts = 8
                single_buffer_configs.append(Config(tiled_mma, bk))
                single_buffer_configs.append(Config(tiled_mma, bk, parallel_k_parts))

        for tiled_mma in _predefined_tiled_mma:
            c_shape, _ = tiled_mma.c_tv_layout()
            bm, bn = c_shape
            if bm <= 32:
                bk = 128
            else:
                bk = 64
            if k >= 2 * n:
                parallel_k_parts = 4
                multi_buffer_configs.append(Config(tiled_mma, bk))
                multi_buffer_configs.append(Config(tiled_mma, bk, parallel_k_parts))
            else:
                parallel_k_parts = 4
                multi_buffer_configs.append(Config(tiled_mma, bk))
                multi_buffer_configs.append(Config(tiled_mma, bk, parallel_k_parts))

        for config in single_buffer_configs:
            self.functions.update({f"single_buffer\n{config}": self._single_buffer(config)})

        for config in multi_buffer_configs:
            self.functions.update({f"multi_buffer\n{config}": self._multi_buffer(config)})

    def _k_partition(self, config: Config):
        _, _, bk = config.thread_block_shape
        parallel_k_parts = config.parallel_k_parts
        k = self.k
        if parallel_k_parts == 1:
            return k

        k_partition = bk
        while k_partition * parallel_k_parts < k:
            k_partition += bk
        return k_partition

    def _single_buffer(self, config: Config):
        m = symbol_var(self.m_symbol_name)
        n = self.n
        k = self.k
        group_size = self.group_size
        wdtype = self.weight_dtype
        bm, bn, bk = config.thread_block_shape
        threads = config.threads
        parallel_k_parts = config.parallel_k_parts
        tiled_mma = config.tiled_mma
        k_tile = config.k_tile

        dynamic_smem_bytes = config.dynamic_smem_bytes(f16, u4, group_size, 1)
        gmem_layout, smem_layout = self._deduce_gmem_layout(bk, bn)
        scale_gmem_layout = TensorLayout((bn, (group_size, k // group_size)), (1, (0, n)))
        if bk > group_size:
            scale_smem_layout = TensorLayout((bn, (group_size, bk // group_size)), (1, (0, bn)))
        else:
            scale_smem_layout = TensorLayout((bn, bk), (1, 0))

        k_partition = self._k_partition(config)

        with hidet.script_module() as script_module:

            @hidet.script
            def func(
                a: f16[m, k],
                b: wdtype[n, k],
                c: f16[m, n],
                scale: f16[k // group_size, n],
                bias: f16[k // group_size, n],
                c_parallel_k_parts: f16[parallel_k_parts, m, n],
                lock: i32[cdiv(m, bn), cdiv(n, bn)],
            ):
                attrs.func_kind = "cuda_kernel"
                attrs.cuda.block_dim = threads
                attrs.cuda.grid_dim = cdiv(m, bm) * cdiv(n, bn), parallel_k_parts
                attrs.cuda.dynamic_smem_bytes = dynamic_smem_bytes

                group_size_m = 8
                pid = blockIdx.x
                num_pid_m = cdiv(m, bm)
                num_pid_n = cdiv(n, bn)
                num_pid_in_group = group_size_m * num_pid_n
                group_id = pid // num_pid_in_group
                first_pid_m = group_id * group_size_m
                group_size_m = min(num_pid_m - first_pid_m, group_size_m)
                pid_m = first_pid_m + (pid % group_size_m)
                pid_n = (pid % num_pid_in_group) // group_size_m

                k_part = blockIdx.y
                if k_part == 0 and threadIdx.x == 0:
                    lock[pid_m, pid_n] = 0

                k_start_pos = k_part * k_partition
                k_start_ofs = gmem_layout((0, k_start_pos))

                ts_a = make_tensor(f16, TensorLayout((bm, bk), (bk, 1)), "shared")
                ts_b = make_tensor(wdtype, smem_layout, "shared")

                ts_scale = make_tensor(f16, scale_smem_layout, "shared")
                ts_bias = make_tensor(f16, scale_smem_layout, "shared")

                tr_a = make_tensor(f16, layout_auto((bm, k_tile * 2)), "register")
                tr_b = make_tensor(wdtype, layout_auto((bn, k_tile * 2)), "register")
                tr_c = make_tensor("float32", auto_layout, "register")
                fill(tr_c, 0.0)

                tr_scale = make_tensor(f16, layout_auto((bn, k_tile * 2), (1, 0)), "register")
                tr_bias = make_tensor(f16, layout_auto((bn, k_tile * 2), (1, 0)), "register")

                tg_a = tensor_view(a[pid_m * bm :, k_start_pos:], TensorLayout((bm, k), (k, 1)), "global")
                tg_b = tensor_view(b[pid_n * bn :, k_start_ofs:], gmem_layout, "global")

                tg_scale = tensor_view(scale[k_start_pos // group_size :, pid_n * bn :], scale_gmem_layout, "global")
                tg_bias = tensor_view(bias[k_start_pos // group_size :, pid_n * bn :], scale_gmem_layout, "global")

                txga = partition_src(tg_a, auto_copy())
                txsa = partition_dst(ts_a, auto_copy())

                txgb = partition_src(tg_b, auto_copy())
                txsb = partition_dst(ts_b, auto_copy())

                txgsc = partition_src(tg_scale, auto_copy())
                txssc = partition_dst(ts_scale, auto_copy())

                txgbi = partition_src(tg_bias, auto_copy())
                txsbi = partition_dst(ts_bias, auto_copy())

                txSa = partition_src(ts_a, auto_copy())
                txra = partition_dst(tr_a, auto_copy())

                txSb = partition_src(ts_b, auto_copy())
                txrb = partition_dst(tr_b, auto_copy())

                txSsc = partition_src(ts_scale, auto_copy())
                txrsc = partition_dst(tr_scale, auto_copy())

                txSbi = partition_src(ts_bias, auto_copy())
                txrbi = partition_dst(tr_bias, auto_copy())

                msk_a = mask(auto_copy(), [m - pid_m * bm, i32(bk)])
                msk_b = mask(auto_copy(), [n - pid_n * bn, i32(bk)])
                msk_scale = mask(auto_copy(), [n - pid_n * bn, i32(bk)])

                cp_async_wait_all()
                syncthreads()

                ksize = k - k_part * k_partition if k_part == parallel_k_parts - 1 else k_partition
                k_block_max = (ksize + bk - 1) // bk
                k_tile_max = bk // k_tile
                for ko in range(k_block_max):
                    copy(auto_copy((bm, bk)), txga[:, :, ko], txsa, msk_a)
                    copy(auto_copy((bn, bk)), txgb[:, :, ko], txsb, msk_b)
                    copy(auto_copy((bn, bk)), txgsc[:, :, ko], txssc, msk_scale)
                    copy(auto_copy((bn, bk)), txgbi[:, :, ko], txsbi, msk_scale)

                    cp_async_wait_all()
                    syncthreads()

                    copy(auto_copy(), txSa[:, :, 0], txra[:, :, 0])
                    copy(auto_copy(), txSb[:, :, 0], txrb[:, :, 0])
                    copy(auto_copy(), txSsc[:, :, 0], txrsc[:, :, 0])
                    copy(auto_copy(), txSbi[:, :, 0], txrbi[:, :, 0])

                    for ki in range(k_tile_max):
                        if ki < k_tile_max - 1:
                            copy(auto_copy(), txSa[:, :, ki + 1], txra[:, :, (ki + 1) % 2])
                            copy(auto_copy(), txSb[:, :, ki + 1], txrb[:, :, (ki + 1) % 2])

                            copy(auto_copy(), txSsc[:, :, ki + 1], txrsc[:, :, (ki + 1) % 2])
                            copy(auto_copy(), txSbi[:, :, ki + 1], txrbi[:, :, (ki + 1) % 2])

                        txrb_f16 = txrsc[:, :, ki % 2] * (cast(txrb[:, :, ki % 2], f16) - txrbi[:, :, ki % 2])
                        mma(tiled_mma, tr_c, txra[:, :, ki % 2], txrb_f16, tr_c)
                    syncthreads()

                k_part = blockIdx.y

                msk_c = mask(auto_copy(), [m - pid_m * bm, n - pid_n * bn])
                tr_c_f16 = cast(tr_c, f16)
                tr_C = rearrange(tr_c_f16, auto_layout, "register")

                lc = ~lock[pid_m, pid_n]
                if k_part < parallel_k_parts - 1:
                    tg_c = tensor_view(
                        c_parallel_k_parts[k_part, pid_m * bm : (pid_m + 1) * bm, pid_n * bn : (pid_n + 1) * bn],
                        TensorLayout((bm, bn), (n, 1)),
                        "global",
                    )

                    txrx_c = partition_src(tr_C, auto_copy())
                    txgx_c = partition_dst(tg_c, auto_copy())
                    copy(auto_copy((bm, bn)), txrx_c, txgx_c, msk_c)

                    syncthreads()
                    if threadIdx.x == 0:
                        atomic_add(lc, 1)
                else:

                    tr_c_k_part = make_tensor("float16", auto_layout, "register")
                    txrx_c_k_part = partition_dst(tr_c_k_part, auto_copy())

                    acquire_seq_semaphore(lc, k_part)

                    for i in range(parallel_k_parts - 1):
                        tg_c = tensor_view(
                            c_parallel_k_parts[i, pid_m * bm : (pid_m + 1) * bm, pid_n * bn : (pid_n + 1) * bn],
                            TensorLayout((bm, bn), (n, 1)),
                            "global",
                        )

                        txgx_c = partition_src(tg_c, auto_copy())
                        copy(auto_copy((bm, bn)), txgx_c, txrx_c_k_part, msk_c)

                        tr_C = tr_c_k_part + tr_C

                    tg_c_final = tensor_view(
                        c[pid_m * bm : (pid_m + 1) * bm, pid_n * bn : (pid_n + 1) * bn],
                        TensorLayout((bm, bn), (n, 1)),
                        "global",
                    )
                    txgx_c_final = partition_dst(tg_c_final, auto_copy())
                    txrx_c_final = partition_src(tr_C, auto_copy())
                    copy(auto_copy((bm, bn)), txrx_c_final, txgx_c_final, msk_c)

        func = script_module.build()
        return func

    def _multi_buffer(self, config: Config):
        m = symbol_var(self.m_symbol_name)
        n = self.n
        k = self.k
        group_size = self.group_size
        wdtype = self.weight_dtype
        bm, bn, bk = config.thread_block_shape
        threads = config.threads
        parallel_k_parts = config.parallel_k_parts
        tiled_mma = config.tiled_mma
        k_tile = config.k_tile

        dynamic_smem_bytes = config.dynamic_smem_bytes(f16, u4, group_size)
        stages = config.stages(f16, u4, group_size)
        gmem_layout, smem_layout = self._deduce_gmem_layout(bk, bn, stages)
        scale_gmem_layout = TensorLayout((bn, (group_size, k // group_size)), (1, (0, n)))
        if bk > group_size:
            scale_smem_layout = TensorLayout(
                (bn, (group_size, bk // group_size), stages), (1, (0, bn), bn * bk // group_size)
            )
        else:
            scale_smem_layout = TensorLayout((bn, bk, stages), (1, 0, bn))

        k_partition = self._k_partition(config)
        assert k_partition >= bk * stages

        with hidet.script_module() as script_module:

            @hidet.script
            def func(
                a: f16[m, k],
                b: wdtype[n, k],
                c: f16[m, n],
                scale: f16[k // group_size, n],
                bias: f16[k // group_size, n],
                c_parallel_k_parts: f16[parallel_k_parts, m, n],
                lock: i32[cdiv(m, bm), cdiv(n, bn)],
            ):
                attrs.func_kind = "cuda_kernel"
                attrs.cuda.block_dim = threads
                attrs.cuda.grid_dim = cdiv(m, bm) * cdiv(n, bn), parallel_k_parts
                attrs.cuda.dynamic_smem_bytes = dynamic_smem_bytes

                group_size_m = 8
                pid = blockIdx.x
                num_pid_m = cdiv(m, bm)
                num_pid_n = cdiv(n, bn)
                num_pid_in_group = group_size_m * num_pid_n
                group_id = pid // num_pid_in_group
                first_pid_m = group_id * group_size_m
                group_size_m = min(num_pid_m - first_pid_m, group_size_m)
                pid_m = first_pid_m + (pid % group_size_m)
                pid_n = (pid % num_pid_in_group) // group_size_m

                k_part = blockIdx.y
                if k_part == 0 and threadIdx.x == 0:
                    lock[pid_m, pid_n] = 0

                k_start_pos = k_part * k_partition
                k_start_ofs = gmem_layout((0, k_start_pos))

                ts_a = make_tensor(f16, TensorLayout((bm, bk, stages), (bk, 1, bm * bk)), "shared")
                ts_b = make_tensor(wdtype, layout_auto((bn, bk, stages)), "shared")

                ts_scale = make_tensor(f16, scale_smem_layout, "shared")
                ts_bias = make_tensor(f16, scale_smem_layout, "shared")

                tr_a = make_tensor(f16, layout_auto((bm, k_tile * 2)), "register")
                tr_b = make_tensor(wdtype, layout_auto((bn, k_tile * 2)), "register")
                tr_c = make_tensor("float32", auto_layout, "register")
                fill(tr_c, 0.0)

                tr_scale = make_tensor(f16, layout_auto((bn, k_tile * 2), (1, 0)), "register")
                tr_bias = make_tensor(f16, layout_auto((bn, k_tile * 2), (1, 0)), "register")

                tg_a = tensor_view(a[pid_m * bm :, k_start_pos:], TensorLayout((bm, k), (k, 1)), "global")
                tg_b = tensor_view(b[pid_n * bn :, k_start_ofs:], gmem_layout, "global")

                tg_scale = tensor_view(scale[k_start_pos // group_size :, pid_n * bn :], scale_gmem_layout, "global")
                tg_bias = tensor_view(bias[k_start_pos // group_size :, pid_n * bn :], scale_gmem_layout, "global")

                txga = partition_src(tg_a, auto_copy())
                txsa = partition_dst(ts_a, auto_copy())

                txgb = partition_src(tg_b, auto_copy())
                txsb = partition_dst(ts_b, auto_copy())

                txgsc = partition_src(tg_scale, auto_copy())
                txssc = partition_dst(ts_scale, auto_copy())

                txgbi = partition_src(tg_bias, auto_copy())
                txsbi = partition_dst(ts_bias, auto_copy())

                msk_a = mask(auto_copy(), [m - pid_m * bm, i32(bk)])
                msk_b = mask(auto_copy(), [n - pid_n * bn, i32(bk)])
                msk_scale = mask(auto_copy(), [n - pid_n * bn, i32(bk)])

                for s in range(stages - 1):
                    copy(auto_copy((bm, bk)), txga[:, :, s], txsa[:, :, s], msk_a)
                    copy(auto_copy((bn, bk)), txgb[:, :, s], txsb[:, :, s], msk_b)
                    copy(auto_copy((bn, bk)), txgsc[:, :, s], txssc[:, :, s], msk_scale)
                    copy(auto_copy((bn, bk)), txgbi[:, :, s], txsbi[:, :, s], msk_scale)
                    cp_async_commit_group()
                cp_async_wait_group(allow_on_fly_groups=stages - 2)
                syncthreads()

                smem_pipe_read = 0
                smem_pipe_write = stages - 1

                txSa = partition_src(ts_a, auto_copy())
                txra = partition_dst(tr_a, auto_copy())

                txSb = partition_src(ts_b, auto_copy())
                txrb = partition_dst(tr_b, auto_copy())

                txSsc = partition_src(ts_scale, auto_copy())
                txrsc = partition_dst(tr_scale, auto_copy())

                txSbi = partition_src(ts_bias, auto_copy())
                txrbi = partition_dst(tr_bias, auto_copy())

                txSa_p = txSa[:, :, :, smem_pipe_read]
                txSb_p = txSb[:, :, :, smem_pipe_read]

                txSsc_p = txSsc[:, :, :, smem_pipe_read]
                txSbi_p = txSbi[:, :, :, smem_pipe_read]

                copy(auto_copy(), txSa_p[:, :, 0], txra[:, :, 0])
                copy(auto_copy(), txSb_p[:, :, 0], txrb[:, :, 0])

                copy(auto_copy(), txSsc_p[:, :, 0], txrsc[:, :, 0])
                copy(auto_copy(), txSbi_p[:, :, 0], txrbi[:, :, 0])

                ksize = k - k_part * k_partition if k_part == parallel_k_parts - 1 else k_partition
                k_block_max = (ksize + bk - 1) // bk
                k_tile_max = bk // k_tile
                for ko in range(k_block_max):
                    for ki in range(k_tile_max):
                        if ki == k_tile_max - 1:
                            # txSa_p = txSa[:, :, :, smem_pipe_read]
                            # txSb_p = txSb[:, :, :, smem_pipe_read]
                            cp_async_wait_group(allow_on_fly_groups=0)
                            syncthreads()

                        k_tile_next = (ki + 1) % k_tile_max
                        copy(auto_copy(), txSa[:, :, k_tile_next, smem_pipe_read], txra[:, :, (ki + 1) % 2])
                        copy(auto_copy(), txSb[:, :, k_tile_next, smem_pipe_read], txrb[:, :, (ki + 1) % 2])
                        # TODO: automate this in compiler pass
                        copy(auto_copy(), txSsc[:, :, k_tile_next, smem_pipe_read], txrsc[:, :, (ki + 1) % 2])
                        copy(auto_copy(), txSbi[:, :, k_tile_next, smem_pipe_read], txrbi[:, :, (ki + 1) % 2])

                        if ki == 0:
                            if ko + stages - 1 < k_block_max:
                                copy(
                                    auto_copy((bm, bk)), txga[:, :, ko + stages - 1], txsa[:, :, smem_pipe_write], msk_a
                                )
                                copy(
                                    auto_copy((bn, bk)), txgb[:, :, ko + stages - 1], txsb[:, :, smem_pipe_write], msk_b
                                )
                                copy(
                                    auto_copy((bn, bk)),
                                    txgsc[:, :, ko + stages - 1],
                                    txssc[:, :, smem_pipe_write],
                                    msk_scale,
                                )
                                copy(
                                    auto_copy((bn, bk)),
                                    txgbi[:, :, ko + stages - 1],
                                    txsbi[:, :, smem_pipe_write],
                                    msk_scale,
                                )
                            smem_pipe_write = smem_pipe_read
                            cp_async_commit_group()

                        if ki == k_tile_max - 2:
                            smem_pipe_read += 1
                            smem_pipe_read = 0 if smem_pipe_read == stages else smem_pipe_read

                        txrb_f16 = txrsc[:, :, ki % 2] * (cast(txrb[:, :, ki % 2], f16) - txrbi[:, :, ki % 2])
                        mma(tiled_mma, tr_c, txra[:, :, ki % 2], txrb_f16, tr_c)

                k_part = blockIdx.y

                msk_c = mask(auto_copy(), [m - pid_m * bm, n - pid_n * bn])
                tr_c_f16 = cast(tr_c, f16)
                tr_C = rearrange(tr_c_f16, auto_layout, "register")

                lc = ~lock[pid_m, pid_n]
                if k_part < parallel_k_parts - 1:

                    tg_c = tensor_view(
                        c_parallel_k_parts[k_part, pid_m * bm : (pid_m + 1) * bm, pid_n * bn : (pid_n + 1) * bn],
                        TensorLayout((bm, bn), (n, 1)),
                        "global",
                    )

                    txrx_c = partition_src(tr_C, auto_copy())
                    txgx_c = partition_dst(tg_c, auto_copy())
                    copy(auto_copy((bm, bn)), txrx_c, txgx_c, msk_c)

                    syncthreads()
                    if threadIdx.x == 0:
                        atomic_add(lc, 1)
                else:

                    tr_c_k_part = make_tensor("float16", auto_layout, "register")
                    txrx_c_k_part = partition_dst(tr_c_k_part, auto_copy())

                    acquire_seq_semaphore(lc, k_part)

                    for i in range(parallel_k_parts - 1):
                        tg_c = tensor_view(
                            c_parallel_k_parts[i, pid_m * bm : (pid_m + 1) * bm, pid_n * bn : (pid_n + 1) * bn],
                            TensorLayout((bm, bn), (n, 1)),
                            "global",
                        )

                        txgx_c = partition_src(tg_c, auto_copy())
                        copy(auto_copy((bm, bn)), txgx_c, txrx_c_k_part, msk_c)

                        tr_C = tr_c_k_part + tr_C

                    tg_c_final = tensor_view(
                        c[pid_m * bm : (pid_m + 1) * bm, pid_n * bn : (pid_n + 1) * bn],
                        TensorLayout((bm, bn), (n, 1)),
                        "global",
                    )
                    txgx_c_final = partition_dst(tg_c_final, auto_copy())
                    txrx_c_final = partition_src(tr_C, auto_copy())
                    copy(auto_copy((bm, bn)), txrx_c_final, txgx_c_final, msk_c)

        func = script_module.build()
        return func


def w4a16_linear(name: str, input_feats: int, output_feats: int, group_size: int):
    return FpAIntBGemm(name, input_feats, output_feats, group_size, u4)
