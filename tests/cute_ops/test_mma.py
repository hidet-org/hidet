import torch
import pytest

import hidet
from hidet.ir.cute.layout import TiledTensorLayout, TensorLayout, make_layout, coalesce
from hidet.ir.cute.layout import ThrValAtom, Level
from hidet.ir.cute.algorithm import CopyAtom, TiledCopy, MmaAtom, TiledMma
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
    fill,
    wgmma_fence_operand,
)
from hidet.lang.mapping import auto_map
from hidet.lang.types import u32, i32, f16, f32, f8e4m3, f8e5m2
from hidet.lang import DataType
from hidet.graph.frontend.torch.utils import dtype_to_torch


@pytest.mark.requires_cuda
def test_mma():
    from hidet.lang import attrs
    from hidet.lang import shared_tensor, register_tensor
    from hidet.lang.cuda import syncthreads, cp_async_wait_all

    a = TensorLayout(((4, 8), (2, 2, 2)), ((32, 1), (16, 8, 128)))
    b = TensorLayout(((4, 8), (2, 2)), ((16, 1), (8, 64)))
    c = TensorLayout(((4, 8), (2, 2)), ((32, 1), (16, 8)))
    mma_atom = MmaAtom("warp", (16, 8, 16), a, b, c, c, (2, 4))
    warp_in_threadblock = Level("warp", "thread_block", (4, 2), TensorLayout((4, 2)), (2, 2))
    tiled_mma = TiledMma(mma_atom, [warp_in_threadblock])
    print(tiled_mma.str_indented())
    a_shape, a_tv_layout = tiled_mma.a_tv_layout()
    b_shape, b_tv_layout = tiled_mma.b_tv_layout()
    c_shape, c_tv_layout = tiled_mma.c_tv_layout()
    print(a_shape, a_tv_layout)
    print(b_shape, b_tv_layout)
    print(c_shape, c_tv_layout)
    m, k = a_shape
    n, k_ = b_shape
    m_, n_ = c_shape
    assert m == m_ and n == n_ and k == k_

    def canonicalize(layout: TensorLayout):
        return layout[0][0], coalesce(make_layout(layout[0][1], layout[1]))

    a_t, a_v = canonicalize(a_tv_layout)
    b_t, b_v = canonicalize(b_tv_layout)
    c_t, c_v = canonicalize(c_tv_layout)

    a_elements = a_v.size()
    b_elements = b_v.size()
    c_elements = c_v.size()

    a_tiled_tensor_layout = TiledTensorLayout(ThrValAtom("thread_block", a_shape, make_layout(a_t, a_v)))
    b_tiled_tensor_layout = TiledTensorLayout(ThrValAtom("thread_block", b_shape, make_layout(b_t, b_v)))
    c_tiled_tensor_layout = TiledTensorLayout(ThrValAtom("thread_block", c_shape, make_layout(c_t, c_v)))

    a_tiled_copy = TiledCopy(CopyAtom.from_tv_atom(a_tiled_tensor_layout.atom))
    b_tiled_copy = TiledCopy(CopyAtom.from_tv_atom(b_tiled_tensor_layout.atom))
    c_tiled_copy = TiledCopy(CopyAtom.from_tv_atom(c_tiled_tensor_layout.atom))

    a_layout = TensorLayout((m, k), (k, 1))
    b_layout = TensorLayout((n, k), (1, n))
    c_layout = TensorLayout((m, n), (n, 1))

    with hidet.script_module() as script_module:

        @hidet.script
        def func(a: f16[m, k], b: f16[k, n], c: f16[m, n]):
            attrs.func_kind = "cuda_kernel"
            attrs.cuda.block_dim = 128
            attrs.cuda.grid_dim = 1

            a_regs = register_tensor("float16", shape=[a_elements])
            b_regs = register_tensor("float16", shape=[b_elements])
            c_regs = register_tensor("float16", shape=[c_elements])

            tr_a = tensor_view(a_regs, a_tiled_tensor_layout, "register")
            tr_b = tensor_view(b_regs, b_tiled_tensor_layout, "register")
            tr_c = tensor_view(c_regs, c_tiled_tensor_layout, "register")

            tg_a = tensor_view(a, a_layout, "global")
            tg_b = tensor_view(b, b_layout, "global")
            txgx_a = partition_src(tg_a, a_tiled_copy)
            txrx_a = partition_dst(tr_a, a_tiled_copy)

            txgx_b = partition_src(tg_b, b_tiled_copy)
            txrx_b = partition_dst(tr_b, b_tiled_copy)

            copy(a_tiled_copy, txgx_a, txrx_a)
            copy(b_tiled_copy, txgx_b, txrx_b)

            mma(tiled_mma, tr_c, txrx_a, txrx_b, tr_c)

            tg_c = tensor_view(c, c_layout, "global")

            txrx_c = partition_src(tr_c, c_tiled_copy)
            txgx_c = partition_dst(tg_c, c_tiled_copy)
            copy(c_tiled_copy, txrx_c, txgx_c)

    func = script_module.build()
    a_mem = hidet.empty([m, k], device="cuda")
    b_mem = hidet.empty([k, n], device="cuda")
    c_mem = hidet.empty([m, n], device="cuda")
    func(a_mem, b_mem, c_mem)


@pytest.mark.requires_cuda
def test_mma_1():
    from hidet.lang.types import u32, i32, f16
    from hidet.lang import attrs
    from hidet.lang import shared_tensor, register_tensor
    from hidet.lang.cuda import syncthreads, cp_async_wait_all

    a = TensorLayout(((4, 8), (2, 2)), ((16, 1), (8, 64)))
    b = TensorLayout(((4, 8), (2, 2, 2)), ((32, 1), (16, 8, 128)))
    c = TensorLayout(((4, 8), (2, 2)), ((2, 8), (1, 64)))
    mma_atom = MmaAtom("warp", (8, 16, 16), a, b, c, c, (4, 1))
    warp_in_threadblock = Level("warp", "thread_block", (2, 2), TensorLayout((2, 2)), (2, 2))
    tiled_mma = TiledMma(mma_atom, [warp_in_threadblock])
    print(tiled_mma.str_indented())
    a_shape, a_tv_layout = tiled_mma.a_tv_layout()
    b_shape, b_tv_layout = tiled_mma.b_tv_layout()
    c_shape, c_tv_layout = tiled_mma.c_tv_layout()
    print(a_shape, a_tv_layout)
    print(b_shape, b_tv_layout)
    print(c_shape, c_tv_layout)
    m, k = a_shape
    n, k_ = b_shape
    m_, n_ = c_shape
    assert m == m_ and n == n_ and k == k_

    def canonicalize(layout: TensorLayout):
        return layout[0][0], coalesce(make_layout(layout[0][1], layout[1]))

    a_t, a_v = canonicalize(a_tv_layout)
    b_t, b_v = canonicalize(b_tv_layout)
    c_t, c_v = canonicalize(c_tv_layout)

    a_elements = a_v.size()
    b_elements = b_v.size()
    c_elements = c_v.size()

    a_tiled_tensor_layout = TiledTensorLayout(ThrValAtom("thread_block", a_shape, make_layout(a_t, a_v)))
    b_tiled_tensor_layout = TiledTensorLayout(ThrValAtom("thread_block", b_shape, make_layout(b_t, b_v)))
    c_tiled_tensor_layout = TiledTensorLayout(ThrValAtom("thread_block", c_shape, make_layout(c_t, c_v)))

    a_tiled_copy = TiledCopy(CopyAtom.from_tv_atom(a_tiled_tensor_layout.atom))
    b_tiled_copy = TiledCopy(CopyAtom.from_tv_atom(b_tiled_tensor_layout.atom))
    c_tiled_copy = TiledCopy(CopyAtom.from_tv_atom(c_tiled_tensor_layout.atom))

    a_layout = TensorLayout((m, k), (k, 1))
    b_layout = TensorLayout((n, k), (1, n))
    c_layout = TensorLayout((m, n), (n, 1))

    with hidet.script_module() as script_module:

        @hidet.script
        def func(a: f16[m, k], b: f16[k, n], c: f16[m, n]):
            attrs.func_kind = "cuda_kernel"
            attrs.cuda.block_dim = 128
            attrs.cuda.grid_dim = 1

            a_regs = register_tensor("float16", shape=[a_elements])
            b_regs = register_tensor("float16", shape=[b_elements])
            c_regs = register_tensor("float16", shape=[c_elements])

            tr_a = tensor_view(a_regs, a_tiled_tensor_layout, "register")
            tr_b = tensor_view(b_regs, b_tiled_tensor_layout, "register")
            tr_c = tensor_view(c_regs, c_tiled_tensor_layout, "register")

            tg_a = tensor_view(a, a_layout, "global")
            tg_b = tensor_view(b, b_layout, "global")
            txgx_a = partition_src(tg_a, a_tiled_copy)
            txrx_a = partition_dst(tr_a, a_tiled_copy)

            txgx_b = partition_src(tg_b, b_tiled_copy)
            txrx_b = partition_dst(tr_b, b_tiled_copy)

            copy(a_tiled_copy, txgx_a, txrx_a)
            copy(b_tiled_copy, txgx_b, txrx_b)

            mma(tiled_mma, tr_c, txrx_a, txrx_b, tr_c)

            tg_c = tensor_view(c, c_layout, "global")

            txrx_c = partition_src(tr_c, c_tiled_copy)
            txgx_c = partition_dst(tg_c, c_tiled_copy)
            copy(c_tiled_copy, txrx_c, txgx_c)

    func = script_module.build()
    a_mem = hidet.empty([m, k], device="cuda")
    b_mem = hidet.empty([k, n], device="cuda")
    c_mem = hidet.empty([m, n], device="cuda")
    func(a_mem, b_mem, c_mem)


def gemm_2(m, n, k):
    from hidet.lang.types import u32, i32, f16
    from hidet.lang import attrs
    from hidet.lang import shared_tensor, register_tensor
    from hidet.lang.cuda import syncthreads, cp_async_wait_all
    from hidet.lang.cuda import blockIdx, threadIdx, dynamic_shared_memory
    from hidet.lang.cuda import cp_async, cp_async_commit_group, cp_async_wait_group

    #    a = TensorLayout(((4, 8), (2, 2)), ((16, 1), (8, 64)))
    #    b = TensorLayout(((4, 8), (2, 2, 2)), ((32, 1), (16, 8, 128)))
    #    c = TensorLayout(((4, 8), (2, 2)), ((2, 8), (1, 64)))
    #    mma_atom = MmaAtom("warp", (8, 16, 16), a, b, c, c, (4, 2))
    #    warp_in_threadblock = Level("warp", "thread_block", (2, 2), TensorLayout((2, 2)), (2, 2))
    #    tiled_mma = TiledMma(mma_atom, [warp_in_threadblock])

    # block_m = 64
    # block_n = 256
    # m % block_m == 0 and n % block_n == 0
    a = TensorLayout(((4, 8), (2, 2, 2)), ((32, 1), (16, 8, 128)))
    b = TensorLayout(((4, 8), (2, 2)), ((16, 1), (8, 64)))
    c = TensorLayout(((4, 8), (2, 2)), ((32, 1), (16, 8)))
    mma_atom = MmaAtom("warp", (16, 8, 16), a, b, c, c, (4, 4))
    warp_in_threadblock = Level("warp", "thread_block", (1, 8), TensorLayout((1, 8)), (1, 1))
    tiled_mma = TiledMma(mma_atom, [warp_in_threadblock])

    #
    #    a = TensorLayout(((4, 8), (2, 2, 2)), ((32, 1), (16, 8, 128)))
    #    b = TensorLayout(((4, 8), (2, 2)), ((16, 1), (8, 64)))
    #    c = TensorLayout(((4, 8), (2, 2)), ((32, 1), (16, 8)))
    #    mma_atom = MmaAtom("warp", (16, 8, 16), a, b, c, c, (2, 4))
    #    warp_in_threadblock = Level("warp", "thread_block", (4, 2), TensorLayout((4, 2)), (2, 2))
    #    tiled_mma = TiledMma(mma_atom, [warp_in_threadblock])

    # a = TensorLayout(((4, 8), (2, 2)), ((16, 1), (8, 64)))
    # b = TensorLayout(((4, 8), (2, 2, 2)), ((32, 1), (16, 8, 128)))
    # c = TensorLayout(((4, 8), (2, 2)), ((2, 8), (1, 64)))
    # mma_atom = MmaAtom("warp", (8, 16, 16), a, b, c, c, (4, 1))
    # warp_in_threadblock = Level("warp", "thread_block", (2, 2), TensorLayout((2, 2)), (2, 2))
    # tiled_mma = TiledMma(mma_atom, [warp_in_threadblock])

    a_shape, a_tv_layout = tiled_mma.a_tv_layout()
    b_shape, b_tv_layout = tiled_mma.b_tv_layout()
    c_shape, c_tv_layout = tiled_mma.c_tv_layout()

    bm, inst_k = a_shape
    bn, inst_k_ = b_shape
    bm_, bn_ = c_shape
    assert bm == bm_ and bn == bn_ and inst_k == inst_k_
    bk = 32

    def canonicalize(layout: TensorLayout):
        return layout[0][0], coalesce(make_layout(layout[0][1], layout[1]))

    a_t, a_v = canonicalize(a_tv_layout)
    b_t, b_v = canonicalize(b_tv_layout)
    c_t, c_v = canonicalize(c_tv_layout)

    a_elements = a_v.size()
    b_elements = b_v.size()
    c_elements = c_v.size()

    threads = c_t.size()

    from hidet.ir.cute.algorithm import auto_copy, auto_mma
    from hidet.ir.cute import layout_auto, auto_layout

    stages = 3
    dynamic_smem_bytes = (bm + bn) * bk * stages * f16.nbytes

    from hidet.utils.py import cdiv
    from hidet.lang import grid
    from hidet.ir.expr import cast

    with hidet.script_module() as script_module:

        @hidet.script
        def func(a: f16[m, k], b: f16[k, n], c: f16[m, n]):
            attrs.func_kind = "cuda_kernel"
            attrs.cuda.block_dim = threads
            attrs.cuda.grid_dim = cdiv(m, bm), cdiv(n, bn), 1
            attrs.cuda.dynamic_smem_bytes = dynamic_smem_bytes

            smem_a = dynamic_shared_memory(byte_offset=0, dtype=f16)
            smem_b = dynamic_shared_memory(byte_offset=stages * bm * bk * f16.nbytes, dtype=f16)

            a_regs = register_tensor("float16", shape=[a_elements, 2])
            b_regs = register_tensor("float16", shape=[b_elements, 2])
            c_regs = register_tensor("float16", shape=[c_elements])

            for i in grid(c_elements):
                c_regs[i] = 0.0

            tg_a = tensor_view(a[blockIdx.x * bm :, :], TensorLayout((bm, k), (k, 1)), "global")
            tg_b = tensor_view(b[:, blockIdx.y * bn :], TensorLayout((bn, k), (1, n)), "global")

            ts_a = tensor_view(smem_a, TensorLayout((bm, bk, stages), (bk, 1, bm * bk)), "shared")
            ts_b = tensor_view(smem_b, TensorLayout((bn, bk, stages), (1, bn, bn * bk)), "shared")

            tr_a = tensor_view(a_regs, layout_auto((bm, inst_k * 2)), "register")
            tr_b = tensor_view(b_regs, layout_auto((bn, inst_k * 2)), "register")
            tr_c = tensor_view(c_regs, auto_layout, "register")

            txga = partition_src(tg_a, auto_copy())
            txsa = partition_dst(ts_a, auto_copy())

            txgb = partition_src(tg_b, auto_copy())
            txsb = partition_dst(ts_b, auto_copy())

            for s in range(stages - 1):
                copy(auto_copy((bm, bk)), txga[:, :, s], txsa[:, :, s])
                copy(auto_copy((bn, bk)), txgb[:, :, s], txsb[:, :, s])
                cp_async_commit_group()
            cp_async_wait_group(allow_on_fly_groups=stages - 2)
            syncthreads()

            smem_pipe_read = 0
            smem_pipe_write = stages - 1

            txSa = partition_src(ts_a, auto_copy())
            txra = partition_dst(tr_a, auto_copy())

            txSb = partition_src(ts_b, auto_copy())
            txrb = partition_dst(tr_b, auto_copy())

            txSa_p = txSa[:, :, :, smem_pipe_read]
            txSb_p = txSb[:, :, :, smem_pipe_read]

            copy(auto_copy(), txSa_p[:, :, 0], txra[:, :, 0])
            copy(auto_copy(), txSb_p[:, :, 0], txrb[:, :, 0])

            k_tile_max = bk // inst_k
            for ko in range((k + bk - 1) // bk):
                for ki in range(k_tile_max):
                    if ki == k_tile_max - 1:
                        # txSa_p = txSa[:, :, :, smem_pipe_read]
                        # txSb_p = txSb[:, :, :, smem_pipe_read]
                        cp_async_wait_group(allow_on_fly_groups=stages - 2)
                        syncthreads()

                    k_tile_next = (ki + 1) % k_tile_max
                    copy(auto_copy(), txSa[:, :, k_tile_next, smem_pipe_read], txra[:, :, (ki + 1) % 2])
                    copy(auto_copy(), txSb[:, :, k_tile_next, smem_pipe_read], txrb[:, :, (ki + 1) % 2])
                    if ki == 0:
                        if ko + stages - 1 < (k + bk - 1) // bk:
                            copy(auto_copy((bm, bk)), txga[:, :, ko + stages - 1], txsa[:, :, smem_pipe_write])
                            copy(auto_copy((bn, bk)), txgb[:, :, ko + stages - 1], txsb[:, :, smem_pipe_write])
                        smem_pipe_write = smem_pipe_read
                        cp_async_commit_group()

                    if ki == k_tile_max - 2:
                        smem_pipe_read += 1
                        smem_pipe_read = 0 if smem_pipe_read == stages else smem_pipe_read

                    mma(tiled_mma, tr_c, txra[:, :, ki % 2], txrb[:, :, ki % 2], tr_c)

            tg_c = tensor_view(
                c[blockIdx.x * bm : (blockIdx.x + 1) * bm, blockIdx.y * bn : (blockIdx.y + 1) * bn],
                TensorLayout((bm, bn), (n, 1)),
                "global",
            )

            tr_C = rearrange(tr_c, auto_layout, "register")

            txrx_c = partition_src(tr_C, auto_copy())
            txgx_c = partition_dst(tg_c, auto_copy())
            copy(auto_copy((bm, bn)), txrx_c, txgx_c)

    func = script_module.build()
    return func


@torch.no_grad()
def bench(functor, args):
    warmup_iters = 10
    bench_iters = 400

    for _ in range(warmup_iters):
        functor(*args)

    latencies = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()

    for _ in range(bench_iters):
        functor(*args)

    end.record()
    end.synchronize()
    latencies.append(start.elapsed_time(end) / bench_iters)
    print(latencies)

    mean = sum(latencies) / len(latencies)
    min_lat = min(latencies)
    max_lat = max(latencies)

    return mean, min_lat, max_lat


def data(M, N, K, generation_dtype: DataType = f16, output_dtype: DataType = f16, device="cuda", return_hidet=False):
    generation_dtype, output_dtype = dtype_to_torch(generation_dtype), dtype_to_torch(output_dtype)
    lo = -2
    hi = 2
    a = torch.randint(low=lo, high=hi, size=(M, K), dtype=generation_dtype, device=device).to(output_dtype)
    # a = torch.ones(M, K, dtype=dtype, device=device)
    b = torch.randint(low=lo, high=hi, size=(K, N), dtype=generation_dtype, device=device).to(output_dtype)
    # b = torch.ones(K, N, dtype=dtype, device=device)
    c = torch.empty((M, N), dtype=generation_dtype, device=device).to(output_dtype)
    if return_hidet:
        a = hidet.from_torch(a)
        b = hidet.from_torch(b)
        c = hidet.from_torch(c)

    return a, b, c


def gemm_multi_buffer(
    m: int,
    n: int,
    k: int,
    tiled_mma: TiledMma,
    bK: int,
    dtype_a: DataType,
    dtype_b: DataType,
    dtype_acc: DataType,
    dtype_output: DataType,
):
    from hidet.lang import attrs
    from hidet.lang import shared_tensor, register_tensor
    from hidet.lang.cuda import syncthreads, cp_async_wait_all
    from hidet.lang.cuda import blockIdx, threadIdx, dynamic_shared_memory
    from hidet.lang.cuda import cp_async, cp_async_commit_group, cp_async_wait_group

    a_shape, a_tv_layout = tiled_mma.a_tv_layout()
    b_shape, b_tv_layout = tiled_mma.b_tv_layout()
    c_shape, c_tv_layout = tiled_mma.c_tv_layout()

    bm, inst_k = a_shape
    bn, inst_k_ = b_shape
    bm_, bn_ = c_shape
    assert bm == bm_ and bn == bn_ and inst_k == inst_k_

    def canonicalize(layout: TensorLayout):
        return layout[0][0], coalesce(make_layout(layout[0][1], layout[1]))

    a_t, a_v = canonicalize(a_tv_layout)
    b_t, b_v = canonicalize(b_tv_layout)
    c_t, c_v = canonicalize(c_tv_layout)

    threads = c_t.size()

    from hidet.ir.cute.algorithm import auto_copy, auto_mma
    from hidet.ir.cute import auto_layout, layout_auto

    assert dtype_a == dtype_b
    stages = 3
    dynamic_smem_bytes = (bm + bn) * bK * stages * dtype_a.nbytes

    from hidet.utils.py import cdiv
    from hidet.lang import grid
    from hidet.ir.expr import cast

    def convert(x):
        return cast(x, dtype_output)

    with hidet.script_module() as script_module:

        @hidet.script
        def func(a: dtype_a[m, k], b: dtype_b[k, n], c: dtype_output[m, n]):
            attrs.func_kind = "cuda_kernel"
            attrs.cuda.block_dim = threads
            attrs.cuda.grid_dim = cdiv(m, bm) * cdiv(n, bn), 1
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

            ts_a = make_tensor(dtype_a.name, TensorLayout((bm, bK, stages), (bK, 1, bm * bK)), "shared")
            ts_b = make_tensor(dtype_b.name, TensorLayout((bn, bK, stages), (1, bn, bn * bK)), "shared")

            tr_a = make_tensor(dtype_a.name, layout_auto((bm, inst_k * 2)), "register")
            tr_b = make_tensor(dtype_b.name, layout_auto((bn, inst_k * 2)), "register")
            tr_c = make_tensor(dtype_acc.name, auto_layout, "register")
            fill(tr_c, 0.0)

            tg_a = tensor_view(a[pid_m * bm :, :], TensorLayout((bm, k), (k, 1)), "global")
            tg_b = tensor_view(b[:, pid_n * bn :], TensorLayout((bn, k), (1, n)), "global")

            txga = partition_src(tg_a, auto_copy())
            txsa = partition_dst(ts_a, auto_copy())

            txgb = partition_src(tg_b, auto_copy())
            txsb = partition_dst(ts_b, auto_copy())

            for s in range(stages - 1):
                copy(auto_copy((bm, bK)), txga[:, :, s], txsa[:, :, s])
                copy(auto_copy((bn, bK)), txgb[:, :, s], txsb[:, :, s])
                cp_async_commit_group()
            cp_async_wait_group(allow_on_fly_groups=stages - 2)
            syncthreads()

            smem_pipe_read = 0
            smem_pipe_write = stages - 1

            txSa = partition_src(ts_a, auto_copy())
            txra = partition_dst(tr_a, auto_copy())

            txSb = partition_src(ts_b, auto_copy())
            txrb = partition_dst(tr_b, auto_copy())

            txSa_p = txSa[:, :, :, smem_pipe_read]
            txSb_p = txSb[:, :, :, smem_pipe_read]

            copy(auto_copy(), txSa_p[:, :, 0], txra[:, :, 0])
            copy(auto_copy(), txSb_p[:, :, 0], txrb[:, :, 0])

            k_tile_max = bK // inst_k
            for ko in range((k + bK - 1) // bK):
                for ki in range(k_tile_max):
                    if ki == k_tile_max - 1:
                        # txSa_p = txSa[:, :, :, smem_pipe_read]
                        # txSb_p = txSb[:, :, :, smem_pipe_read]
                        cp_async_wait_group(allow_on_fly_groups=stages - 2)
                        syncthreads()

                    k_tile_next = (ki + 1) % k_tile_max
                    copy(auto_copy(), txSa[:, :, k_tile_next, smem_pipe_read], txra[:, :, (ki + 1) % 2])
                    copy(auto_copy(), txSb[:, :, k_tile_next, smem_pipe_read], txrb[:, :, (ki + 1) % 2])
                    if ki == 0:
                        if ko + stages - 1 < (k + bK - 1) // bK:
                            copy(auto_copy((bm, bK)), txga[:, :, ko + stages - 1], txsa[:, :, smem_pipe_write])
                            copy(auto_copy((bn, bK)), txgb[:, :, ko + stages - 1], txsb[:, :, smem_pipe_write])
                        smem_pipe_write = smem_pipe_read
                        cp_async_commit_group()

                    if ki == k_tile_max - 2:
                        smem_pipe_read += 1
                        smem_pipe_read = 0 if smem_pipe_read == stages else smem_pipe_read

                    mma(tiled_mma, tr_c, txra[:, :, ki % 2], txrb[:, :, ki % 2], tr_c)

            tg_c = tensor_view(
                c[pid_m * bm : (pid_m + 1) * bm, pid_n * bn : (pid_n + 1) * bn],
                TensorLayout((bm, bn), (n, 1)),
                "global",
            )

            tr_c_output_dtype = arithmetic(tr_c, op=convert)

            tr_C = rearrange(tr_c_output_dtype, auto_layout, "register")

            txrx_c = partition_src(tr_C, auto_copy())
            txgx_c = partition_dst(tg_c, auto_copy())
            copy(auto_copy((bm, bn)), txrx_c, txgx_c)

    func = script_module.build()
    return func


def gemm_single_buffer(m, n, k):
    from hidet.lang.types import u32, i32, f16, f32
    from hidet.lang import attrs
    from hidet.lang import shared_tensor, register_tensor
    from hidet.lang.cuda import syncthreads, cp_async_wait_all
    from hidet.lang.cuda import blockIdx, threadIdx, dynamic_shared_memory
    from hidet.lang.cuda import cp_async, cp_async_commit_group, cp_async_wait_group

    # a = TensorLayout(((4, 8), (2, 2)), ((16, 1), (8, 64)))
    # b = TensorLayout(((4, 8), (2, 2, 2)), ((32, 1), (16, 8, 128)))
    # c = TensorLayout(((4, 8), (2, 2)), ((2, 8), (1, 64)))
    # mma_atom = MmaAtom("warp", (8, 16, 16), a, b, c, c, (1, 2))
    # warp_in_threadblock = Level("warp", "thread_block", (3, 2), TensorLayout((3, 2)), (1, 2))
    # tiled_mma = TiledMma(mma_atom, [warp_in_threadblock])

    # block_m = 32
    # block_n = 32
    # m % block_m == 0 and n % block_n == 0
    a = TensorLayout(((4, 8), (2, 2, 2)), ((32, 1), (16, 8, 128)))
    b = TensorLayout(((4, 8), (2, 2)), ((16, 1), (8, 64)))
    c = TensorLayout(((4, 8), (2, 2)), ((32, 1), (16, 8)))
    mma_atom = MmaAtom("warp", (16, 8, 16), a, b, c, c, (1, 2))
    warp_in_threadblock = Level("warp", "thread_block", (2, 2), TensorLayout((2, 2)), (1, 1))
    tiled_mma = TiledMma(mma_atom, [warp_in_threadblock])

    #    a = TensorLayout(((4, 8), (2, 2, 2)), ((32, 1), (16, 8, 128)))
    #    b = TensorLayout(((4, 8), (2, 2)), ((16, 1), (8, 64)))
    #    c = TensorLayout(((4, 8), (2, 2)), ((32, 1), (16, 8)))
    #    mma_atom = MmaAtom("warp", (16, 8, 16), a, b, c, c, (2, 8))
    #    warp_in_threadblock = Level("warp", "thread_block", (2, 4), TensorLayout((2, 4)), (1, 1))
    #    tiled_mma = TiledMma(mma_atom, [warp_in_threadblock])

    # a = TensorLayout(((4, 8), (2, 2)), ((16, 1), (8, 64)))
    # b = TensorLayout(((4, 8), (2, 2, 2)), ((32, 1), (16, 8, 128)))
    # c = TensorLayout(((4, 8), (2, 2)), ((2, 8), (1, 64)))
    # mma_atom = MmaAtom("warp", (8, 16, 16), a, b, c, c, (4, 1))
    # warp_in_threadblock = Level("warp", "thread_block", (2, 2), TensorLayout((2, 2)), (2, 2))
    # tiled_mma = TiledMma(mma_atom, [warp_in_threadblock])

    a_shape, a_tv_layout = tiled_mma.a_tv_layout()
    b_shape, b_tv_layout = tiled_mma.b_tv_layout()
    c_shape, c_tv_layout = tiled_mma.c_tv_layout()

    bm, inst_k = a_shape
    bn, inst_k_ = b_shape
    bm_, bn_ = c_shape
    assert bm == bm_ and bn == bn_ and inst_k == inst_k_
    bk = 32

    def canonicalize(layout: TensorLayout):
        return layout[0][0], coalesce(make_layout(layout[0][1], layout[1]))

    a_t, a_v = canonicalize(a_tv_layout)
    b_t, b_v = canonicalize(b_tv_layout)
    c_t, c_v = canonicalize(c_tv_layout)

    threads = c_t.size()

    a_elements = a_v.size()
    b_elements = b_v.size()
    c_elements = c_v.size()

    from hidet.ir.cute.algorithm import auto_copy, auto_mma
    from hidet.ir.cute import auto_layout
    from hidet.ir.cute import layout_auto

    stages = 1
    dynamic_smem_bytes = (bm + bn) * bk * stages * f16.nbytes

    from hidet.utils.py import cdiv
    from hidet.lang import grid
    from hidet.ir.expr import cast

    def convert(x):
        return cast(x, f16)

    with hidet.script_module() as script_module:

        @hidet.script
        def func(a: f16[m, k], b: f16[k, n], c: f16[m, n]):
            attrs.func_kind = "cuda_kernel"
            attrs.cuda.block_dim = threads
            attrs.cuda.grid_dim = cdiv(m, bm) * cdiv(n, bn), 1
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

            smem_a = dynamic_shared_memory(byte_offset=0, dtype=f16)
            smem_b = dynamic_shared_memory(byte_offset=stages * bm * bk * f16.nbytes, dtype=f16)

            a_regs = register_tensor("float16", shape=[a_elements, 2])
            b_regs = register_tensor("float16", shape=[b_elements, 2])
            c_regs = register_tensor("float32", shape=[c_elements])

            for i in grid(c_elements):
                c_regs[i] = 0.0

            tg_a = tensor_view(a[pid_m * bm :, :], TensorLayout((bm, k), (k, 1)), "global")
            tg_b = tensor_view(b[:, pid_n * bn :], TensorLayout((bn, k), (1, n)), "global")

            ts_a = tensor_view(smem_a, TensorLayout((bm, bk), (bk, 1)), "shared")
            ts_b = tensor_view(smem_b, TensorLayout((bn, bk), (1, bn)), "shared")

            tr_a = tensor_view(a_regs, layout_auto((bm, inst_k * 2)), "register")
            tr_b = tensor_view(b_regs, layout_auto((bn, inst_k * 2)), "register")
            tr_c = tensor_view(c_regs, auto_layout, "register")

            txga = partition_src(tg_a, auto_copy())
            txsa = partition_dst(ts_a, auto_copy())

            txgb = partition_src(tg_b, auto_copy())
            txsb = partition_dst(ts_b, auto_copy())

            txSa = partition_src(ts_a, auto_copy())
            txra = partition_dst(tr_a, auto_copy())

            txSb = partition_src(ts_b, auto_copy())
            txrb = partition_dst(tr_b, auto_copy())

            k_tile_max = bk // inst_k
            for ko in range((k + bk - 1) // bk):
                copy(auto_copy((bm, bk)), txga[:, :, ko], txsa)
                copy(auto_copy((bn, bk)), txgb[:, :, ko], txsb)
                cp_async_wait_all()
                syncthreads()

                copy(auto_copy(), txSa[:, :, 0], txra[:, :, 0])
                copy(auto_copy(), txSb[:, :, 0], txrb[:, :, 0])

                for ki in range(k_tile_max):
                    if ki < k_tile_max - 1:
                        copy(auto_copy(), txSa[:, :, ki + 1], txra[:, :, (ki + 1) % 2])
                        copy(auto_copy(), txSb[:, :, ki + 1], txrb[:, :, (ki + 1) % 2])

                    mma(tiled_mma, tr_c, txra[:, :, ki % 2], txrb[:, :, ki % 2], tr_c)
                syncthreads()

            tg_c = tensor_view(
                c[pid_m * bm : (pid_m + 1) * bm, pid_n * bn : (pid_n + 1) * bn],
                TensorLayout((bm, bn), (n, 1)),
                "global",
            )

            tr_c_f16 = arithmetic(tr_c, op=convert)

            tr_C = rearrange(tr_c_f16, auto_layout, "register")

            txrx_c = partition_src(tr_C, auto_copy())
            txgx_c = partition_dst(tg_c, auto_copy())
            copy(auto_copy((bm, bn)), txrx_c, txgx_c)

    func = script_module.build()
    # a_mem = hidet.empty([m, k], device="cuda")
    # b_mem = hidet.empty([k, n], device="cuda")
    # c_mem = hidet.empty([m, n], device="cuda")
    # func(a_mem, b_mem, c_mem)
    return func


@pytest.mark.requires_cuda
@pytest.mark.parametrize("m,n,k", [(4096, 4096, 4096), (120000, 320, 768)])
def test_gemm_single_buffer(m, n, k):
    # hidet.option.cache_dir("./demo_mma")
    # hidet.option.search_space(2)
    # hidet.option.debug_cache_tuning()
    # hidet.option.save_lower_ir(True)

    func = gemm_single_buffer(m, n, k)
    a, b, c = data(m, n, k, return_hidet=True)
    mean, min_lat, max_lat = bench(func, (a, b, c))
    print("time={:.3f} ms, performance={:.3f} TFLOPS".format(mean, 2.0 * (m * n * k) / (1e9 * mean)))

    def fn():
        func(a, b, c)

    from hidet.utils.benchmark import do_bench

    mean = do_bench(fn, percentiles=None)
    print("time={:.3f} ms, performance={:.3f} TFLOPS".format(mean, 2.0 * (m * n * k) / (1e9 * mean)))

    a = a.torch()
    b = b.torch()

    def fn():
        c = a @ b

    from hidet.utils.benchmark import do_bench

    mean, min_lat, max_lat = bench(fn, ())
    print("time={:.3f} ms, performance={:.3f} TFLOPS".format(mean, 2.0 * (m * n * k) / (1e9 * mean)))

    c2 = a @ b
    import numpy as np

    np.set_printoptions(threshold=3000, linewidth=200, edgeitems=100)
    np.testing.assert_allclose(actual=c.cpu().numpy(), desired=c2.cpu().numpy(), rtol=1e-2)


@pytest.mark.requires_cuda
@pytest.mark.parametrize("gemm", [gemm_2, gemm_multi_buffer])
@pytest.mark.parametrize("m,n,k", [(4096, 4096, 4096), (120000, 768, 320)])
def test_gemm_multi_buffer_f16(gemm, m, n, k):
    hidet.option.cache_dir("./demo_mma")
    hidet.option.search_space(2)
    hidet.option.debug_cache_tuning()
    hidet.option.save_lower_ir(True)

    # m16n8k16.f32.f16.f16.f32 TiledMMA
    dtype_a, dtype_b, dtype_acc, dtype_output = f16, f16, f32, f16
    bK = 32
    a = TensorLayout(((4, 8), (2, 2, 2)), ((32, 1), (16, 8, 128)))
    b = TensorLayout(((4, 8), (2, 2)), ((16, 1), (8, 64)))
    c = TensorLayout(((4, 8), (2, 2)), ((32, 1), (16, 8)))
    mma_atom = MmaAtom("warp", (16, 8, 16), a, b, c, c, (1, 2))
    warp_in_threadblock = Level("warp", "thread_block", (2, 2), TensorLayout((2, 2)), (4, 4))
    tiled_mma = TiledMma(mma_atom, [warp_in_threadblock])

    func = (
        gemm(m, n, k, tiled_mma, bK, dtype_a, dtype_b, dtype_acc, dtype_output)
        if gemm == gemm_multi_buffer
        else gemm(m, n, k)
    )
    a, b, c = data(m, n, k, return_hidet=True)
    mean, min_lat, max_lat = bench(func, (a, b, c))
    print("time={:.3f} ms, performance={:.3f} TFLOPS".format(mean, 2.0 * (m * n * k) / (1e9 * mean)))

    def fn():
        func(a, b, c)

    from hidet.utils.benchmark import do_bench

    mean = do_bench(fn, percentiles=None)
    print("time={:.3f} ms, performance={:.3f} TFLOPS".format(mean, 2.0 * (m * n * k) / (1e9 * mean)))

    a = a.torch()
    b = b.torch()

    def fn():
        c = a @ b

    from hidet.utils.benchmark import do_bench

    mean, min_lat, max_lat = bench(fn, ())
    print("time={:.3f} ms, performance={:.3f} TFLOPS".format(mean, 2.0 * (m * n * k) / (1e9 * mean)))

    c2 = a @ b
    import numpy as np

    np.set_printoptions(threshold=3000, linewidth=200, edgeitems=100)
    np.testing.assert_allclose(actual=c.cpu().numpy(), desired=c2.cpu().numpy(), rtol=1e-2)


@pytest.mark.requires_cuda
@pytest.mark.parametrize("gemm", [gemm_multi_buffer])
@pytest.mark.parametrize("f8type", [f8e4m3])
@pytest.mark.parametrize("m,n,k", [(4096, 4096, 4096), (120000, 768, 320)])
def test_gemm_multi_buffer_f8(gemm, f8type, m, n, k):
    # TODO: test f8e5m2 matmul. Multiplication of two Float8_e5m2 matrices is not supported in PyTorch.
    hidet.option.cache_dir("./demo_mma")
    hidet.option.search_space(2)
    hidet.option.debug_cache_tuning()
    hidet.option.save_lower_ir(True)

    device = "cuda"
    atol = 1e-2
    rtol = 1e-2

    # m16n8k32.f32.e4m3.e4m3.f32 TiledMMA
    dtype_a, dtype_b, dtype_acc, dtype_output = f8type, f8type, f32, f8type
    bK = 64  # since inst_k==32
    a = TensorLayout(((4, 8), (4, 2, 2)), ((64, 1), (16, 8, 256)))  # M-major indexing
    b = TensorLayout(((4, 8), (4, 2)), ((32, 1), (8, 128)))  # N-major indexing
    c = TensorLayout(((4, 8), (2, 2)), ((32, 1), (16, 8)))  # M-major indexing
    mma_atom = MmaAtom("warp", (16, 8, 32), a, b, c, c, (1, 2))
    warp_in_threadblock = Level("warp", "thread_block", (2, 2), TensorLayout((2, 2)), (4, 4))
    tiled_mma = TiledMma(mma_atom, [warp_in_threadblock])

    func = gemm(m, n, k, tiled_mma, bK, dtype_a, dtype_b, dtype_acc, dtype_output)
    a, b, c = data(m, n, k, output_dtype=dtype_a, device=device, return_hidet=True)
    mean, min_lat, max_lat = bench(func, (a, b, c))
    print("time={:.3f} ms, performance={:.3f} TFLOPS".format(mean, 2.0 * (m * n * k) / (1e9 * mean)))

    def fn():
        func(a, b, c)

    from hidet.utils.benchmark import do_bench

    mean = do_bench(fn, percentiles=None)
    print("time={:.3f} ms, performance={:.3f} TFLOPS".format(mean, 2.0 * (m * n * k) / (1e9 * mean)))

    a = a.torch()
    b = b.transpose(0, 1).torch()

    def fn(mat1, mat2):
        c = torch._scaled_mm(
            mat1,
            mat2,
            out_dtype=dtype_to_torch(dtype_output),
            scale_a=torch.tensor(1.0, device=device),
            scale_b=torch.tensor(1.0, device=device),
        )
        return c

    from hidet.utils.benchmark import do_bench

    mean, min_lat, max_lat = bench(fn, (a, b.T))
    print("time={:.3f} ms, performance={:.3f} TFLOPS".format(mean, 2.0 * (m * n * k) / (1e9 * mean)))

    c2 = fn(a, b.T)
    torch.testing.assert_close(
        actual=c.torch().to(dtype=torch.float16), expected=c2.to(dtype=torch.float16), atol=atol, rtol=rtol
    )


@pytest.mark.requires_cuda
def test_wgmma_fence_operand():
    from hidet.lang.types import f32
    from hidet.lang import attrs
    from hidet.ir.cute import auto_copy
    from hidet.ir.cute import auto_layout

    m = 128
    n = 128
    with hidet.script_module() as script_module:

        @hidet.script
        def func(tensor: f32[m, n], out: f32[m, n]):
            attrs.func_kind = "cuda_kernel"
            attrs.cuda.block_dim = 128
            attrs.cuda.grid_dim = 1

            tr = make_tensor("float32", auto_layout, "register")
            txrx = partition_dst(tr, auto_copy())
            tg = tensor_view(tensor, TensorLayout((m, n), (n, 1)), "global")
            txgx = partition_src(tg, auto_copy())
            copy(auto_copy((m, n)), txgx, txrx)

            wgmma_fence_operand(tr)

            tg_out = tensor_view(out, TensorLayout((m, n), (n, 1)), "global")
            txrx1 = partition_src(tr, auto_copy())
            txgout = partition_dst(tg_out, auto_copy())
            copy(auto_copy((m, n)), txrx1, txgout)

    func = script_module.build()
    tensor = hidet.empty([m, n], device="cuda")
    out = hidet.empty([m, n], device="cuda")
    func(tensor, out)
