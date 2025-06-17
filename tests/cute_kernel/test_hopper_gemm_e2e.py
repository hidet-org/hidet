from hidet.ir.cute.contexts import warp_groups_consumer, warp_groups_producer
import torch
import pytest

from hidet.lang.types import f8e4m3, f8e5m2, f16, f32
from hidet.lang import attrs, grid
from hidet.lang.cuda import blockIdx, threadIdx
from hidet.ir.primitives.cuda.wgmma import wgmma_fence, wgmma_commit_group, wgmma_wait_group
from hidet.lang.cuda import syncthreads, cp_async_commit_group, cp_async_wait_group
from hidet.lang.constructs.declare import as_tensor_pointer
from hidet.utils.py import cdiv

import hidet
from hidet.ir.cute.algorithm import MmaAtom, TiledMma, auto_copy
from hidet.ir.cute.layout import TensorLayout, Level
from hidet.ir.cute import layout_auto, auto_layout
from hidet.ir.cute.algorithm import CopyAtom, TiledCopy

from hidet.ir.cute.ops import (
    make_tensor,
    tensor_view,
    partition_src,
    partition_dst,
    partition_A,
    partition_B,
    copy,
    mma,
    rearrange,
    cast,
    fill,
    make_mbarriers,
    mbarrier_arrive,
    mbarrier_try_wait,
    mbarrier_wait,
    wgmma_fence_operand,
    transpose,
    mask,
)
from hidet.utils.benchmark import do_bench


def data(M, N, K, trans_a=False, trans_b=False, dtype="float16", device="cuda", return_hidet=False):
    dtype = getattr(torch, dtype)
    lo = -3
    hi = 3
    shape_a = (K, M) if trans_a else (M, K)
    shape_b = (N, K) if trans_b else (K, N)
    a = torch.randint(low=lo, high=hi, size=shape_a, dtype=dtype, device=device)
    b = torch.randint(low=lo, high=hi, size=shape_b, dtype=dtype, device=device)
    c = torch.empty((M, N), dtype=dtype, device=device)

    if return_hidet:
        a = hidet.from_torch(a)
        b = hidet.from_torch(b)
        c = hidet.from_torch(c)

    return a, b, c


def f8_quant_data(M, N, K, trans_a=False, trans_b=False, dtype="int8", device="cuda", return_hidet=False, group_k=64):
    dtype = getattr(torch, dtype)
    lo = -3
    hi = 3
    shape_a = (K, M) if trans_a else (M, K)
    shape_b = (N, K) if trans_b else (K, N)
    a = torch.randint(low=lo, high=hi, size=shape_a, dtype=torch.int8, device=device)
    b = torch.randint(low=lo, high=hi, size=shape_b, dtype=torch.int8, device=device)
    scale_a = torch.randint(low=lo, high=hi, size=(M, K // group_k), dtype=torch.float32, device=device)
    scale_b = torch.randint(low=lo, high=hi, size=(N, K // group_k), dtype=torch.float32, device=device)
    c = torch.empty((M, N), dtype=dtype, device=device)

    if return_hidet:
        a = hidet.from_torch(a)
        b = hidet.from_torch(b)
        scale_a = hidet.from_torch(scale_a)
        scale_b = hidet.from_torch(scale_b)
        c = hidet.from_torch(c)

    return a, b, scale_a, scale_b, c


def f8_gemm_multiple_stage_ss(m, n, k, wgmma_n=64, trans_b=True, group_k=128):
    a = TensorLayout(((128,), (64, 32)), ((0,), (1, 64)))
    b = TensorLayout(((128,), (wgmma_n, 32)), ((0,), (1, wgmma_n)))
    c = TensorLayout(((4, 8, 4), (2, 2, wgmma_n // 8)), ((128, 1, 16), (64, 8, 512)))
    mma_atom = MmaAtom("warp_group", (64, wgmma_n, 32), a, b, c, c)
    wg_in_threadblock = Level("warp_group", "thread_block", (2, 1), TensorLayout((2, 1)), (1, 1))
    tiled_mma = TiledMma(mma_atom, [wg_in_threadblock])

    a_shape, a_tv = tiled_mma.a_tv_layout()
    b_shape, b_tv = tiled_mma.b_tv_layout()
    c_shape, c_tv = tiled_mma.c_tv_layout()
    from quant_utils import canonicalize

    a_t, _ = canonicalize(a_tv)
    _, _ = canonicalize(b_tv)
    _, _ = canonicalize(c_tv)

    bm, inst_k = a_shape
    print(inst_k)
    bn, inst_k_ = b_shape
    bm_, bn_ = c_shape
    assert bm == bm_ and bn == bn_ and inst_k == inst_k_
    threads = a_t.size()
    bk = group_k
    tma_copy_tx = (bm * bk + bn * bk) * f8e4m3.nbytes + (bm + (bn // group_k)) * f32.nbytes
    k_pipe_max = 5

    num_consumer_threads = threads
    num_producer_threads = 128

    with hidet.script_module() as script_module:

        @hidet.script
        def func(
            a: f8e4m3[m, k],
            b_ptr: ~f8e4m3,
            c: f8e4m3[m, n],
            scale_a: f32[m, k // group_k],
            scale_b: f32[n // group_k, k // group_k],
        ):
            attrs.func_kind = "cuda_kernel"
            attrs.cuda.block_dim = num_producer_threads + num_consumer_threads  # 8 warps
            attrs.cuda.grid_dim = cdiv(m, bm), cdiv(n, bn), 1
            attrs.cuda.min_blocks = 1
            attrs.cuda.dynamic_smem_bytes = 0

            bid_x = blockIdx.x
            bid_y = blockIdx.y

            mbar_tma = make_mbarriers(k_pipe_max)
            mbar_mma = make_mbarriers(k_pipe_max)
            tg_a = tensor_view(a, TensorLayout((m, k), (k, 1)), "global", (bm, k), (bid_x * bm, 0))
            b = as_tensor_pointer(b_ptr, f8e4m3, [n, k])
            tg_b = tensor_view(b, TensorLayout((n, k), (k, 1)), "global", (bn, k), (bid_y * bn, 0))
            tg_sa = tensor_view(
                scale_a,
                TensorLayout((m, (bn, k // group_k)), (k // group_k, (0, 1))),
                "global",
                (bm, bn * k // group_k),
                (bid_x * bm, 0),
            )
            tg_sb = tensor_view(
                scale_b,
                TensorLayout(((group_k, n // group_k), (bm, k // group_k)), ((0, k // group_k), (0, 1))),
                "global",
                (bn, bm * k // group_k),
                (bid_y * bn, 0),
            )

            ts_sb = make_tensor("float32", TensorLayout((bn, bm, k_pipe_max), (0, 0, 1)), "shared")
            ts_sa = make_tensor("float32", TensorLayout((bm, bn, k_pipe_max), (1, 0, bm)), "shared")
            ts_b = make_tensor(f8e4m3, layout_auto((bn, bk, k_pipe_max)), "shared")
            ts_a = make_tensor(f8e4m3, layout_auto((bm, bk, k_pipe_max)), "shared")

            syncthreads()

            with warp_groups_producer([2], num_regs=40):
                smem_pipe_write = 0
                write_phase = True

                txga = partition_src(tg_a, auto_copy())
                txsa = partition_dst(ts_a, auto_copy())
                txgb = partition_src(tg_b, auto_copy())
                txsb = partition_dst(ts_b, auto_copy())
                txgsa = partition_src(tg_sa, auto_copy())
                txgsb = partition_src(tg_sb, auto_copy())
                txssa = partition_dst(ts_sa, auto_copy())
                txssb = partition_dst(ts_sb, auto_copy())

                k_blocks = cdiv(k, bk)
                for ko in grid(k_blocks, attrs="u5"):
                    if ko >= k_pipe_max:
                        mbarrier_wait(mbar_mma[smem_pipe_write], write_phase)
                    copy(
                        auto_copy((bm, bk)),
                        txga[:, :, ko],
                        txsa[:, :, smem_pipe_write],
                        mbarrier=mbar_tma[smem_pipe_write],
                    )
                    copy(
                        auto_copy((bn, bk)),
                        txgb[:, :, ko],
                        txsb[:, :, smem_pipe_write],
                        mbarrier=mbar_tma[smem_pipe_write],
                    )
                    copy(
                        auto_copy((bm, bn)),
                        txgsa[:, :, ko],
                        txssa[:, :, smem_pipe_write],
                        mbarrier=mbar_tma[smem_pipe_write],
                    )
                    copy(
                        auto_copy((bn, bm)),
                        txgsb[:, :, ko],
                        txssb[:, :, smem_pipe_write],
                        mbarrier=mbar_tma[smem_pipe_write],
                    )

                    mbarrier_arrive(mbar_tma[smem_pipe_write], tma_copy_tx)
                    smem_pipe_write += 1
                    if smem_pipe_write == k_pipe_max:
                        smem_pipe_write = 0
                        write_phase = not write_phase

            with warp_groups_consumer([0, 1], num_regs=232):
                smem_pipe_read = 0
                read_phase = False
                smem_pipe_release = 0
                release_phase = False

                tr_c_final = make_tensor("float32", layout_auto((bm, bn)), "register")
                tr_sa = make_tensor("float32", layout_auto((bm, bn), (1, 0)), "register")
                tr_sb = make_tensor("float32", layout_auto((bm, bn), (0, 0)), "register")
                ts_sbt = transpose(ts_sb, 1, 0, 2)

                fill(tr_c_final, 0.0)

                txSa = partition_A(ts_a, tiled_mma)
                txSb = partition_B(ts_b, tiled_mma)

                txSsa = partition_src(ts_sa, auto_copy())
                txSsb = partition_src(ts_sbt, auto_copy())
                txrsa = partition_dst(tr_sa, auto_copy())
                txrsb = partition_dst(tr_sb, auto_copy())

                k_blocks = cdiv(k, bk)
                k_tiles = cdiv(bk, inst_k)

                for ko in grid(k_blocks, attrs="u5"):
                    tr_c = make_tensor("float32", layout_auto((bm, bn)), "register")
                    mbarrier_wait(mbar_tma[smem_pipe_read], read_phase)
                    fill(tr_c, 0.0)
                    copy(auto_copy(), txSsa[:, :, smem_pipe_read], txrsa)
                    copy(auto_copy(), txSsb[:, :, smem_pipe_read], txrsb)
                    scale = txrsa * txrsb
                    wgmma_fence_operand(tr_c)
                    wgmma_fence()
                    for ki in range(k_tiles):
                        mma(tiled_mma, tr_c, txSa[:, :, ki, smem_pipe_read], txSb[:, :, ki, smem_pipe_read], tr_c)
                    wgmma_commit_group()
                    wgmma_fence_operand(tr_c)
                    wgmma_wait_group(0)
                    mbarrier_arrive(mbar_mma[smem_pipe_release])
                    tr_c_final = tr_c * scale + tr_c_final

                    smem_pipe_read += 1
                    if smem_pipe_read == k_pipe_max:
                        smem_pipe_read = 0
                        read_phase = not read_phase
                    smem_pipe_release += 1
                    if smem_pipe_release == k_pipe_max:
                        smem_pipe_release = 0
                        release_phase = not release_phase

                tr_C = rearrange(cast(tr_c_final, f8e4m3), auto_layout, "register")

                tg_c = tensor_view(
                    c[bid_x * bm : (bid_x + 1) * bm, bid_y * bn : (bid_y + 1) * bn],
                    TensorLayout((bm, bn), (n, 1)),
                    "global",
                )
                txgc = partition_src(tg_c, auto_copy())
                txrc = partition_dst(tr_C, auto_copy())
                copy(auto_copy((bm, bn)), txrc, txgc)

    func = script_module.build()
    return func


def gemm_multiple_stage_rs(m, n, k, wgmma_n=64, trans_b=True):
    a = TensorLayout(((4, 8, 4), (2, 2, 2)), ((128, 1, 16), (64, 8, 512)))
    b = TensorLayout(((128,), (wgmma_n, 16)), ((0,), (1, wgmma_n)))
    c = TensorLayout(((4, 8, 4), (2, 2, wgmma_n // 8)), ((128, 1, 16), (64, 8, 512)))
    mma_atom = MmaAtom("warp_group", (64, wgmma_n, 16), a, b, c, c)
    wg_in_threadblock = Level("warp_group", "thread_block", (2, 1), TensorLayout((2, 1)), (1, 1))
    tiled_mma = TiledMma(mma_atom, [wg_in_threadblock])

    a_shape, a_tv = tiled_mma.a_tv_layout()
    b_shape, b_tv = tiled_mma.b_tv_layout()
    c_shape, c_tv = tiled_mma.c_tv_layout()
    from quant_utils import canonicalize

    a_t, _ = canonicalize(a_tv)
    _, _ = canonicalize(b_tv)
    _, _ = canonicalize(c_tv)

    bm, inst_k = a_shape
    bn, inst_k_ = b_shape
    bm_, bn_ = c_shape
    assert bm == bm_ and bn == bn_ and inst_k == inst_k_
    threads = a_t.size()
    bk = 64
    tma_copy_tx = (bm * bk + bn * bk) * f16.nbytes
    k_pipe_max = 4
    k_pipe_mmas = 2

    copy_atom = CopyAtom("thread_block", (bm, bk), TensorLayout(((threads,), (bm, bk)), ((0,), (1, bm))))
    tiled_copy_a = TiledCopy(copy_atom, [])

    copy_atom = CopyAtom("thread_block", (bn, bk), TensorLayout(((threads,), (bn, bk)), ((0,), (1, bn))))
    tiled_copy_b = TiledCopy(copy_atom, [])

    with hidet.script_module() as script_module:

        @hidet.script
        def func(a: f16[m, k], b_ptr: ~f16, c: f16[m, n]):
            attrs.func_kind = "cuda_kernel"
            attrs.cuda.block_dim = threads  # 8 warps
            attrs.cuda.grid_dim = cdiv(m, bm), cdiv(n, bn), 1
            attrs.cuda.dynamic_smem_bytes = 0

            bid_x = blockIdx.x
            bid_y = blockIdx.y

            mbar_tma = make_mbarriers(k_pipe_max)
            mbar_mma = make_mbarriers(k_pipe_max)
            tg_a = tensor_view(a, TensorLayout((m, k), (k, 1)), "global", (bm, k), (bid_x * bm, 0))
            b = as_tensor_pointer(b_ptr, "float16", [n, k])
            tg_b = tensor_view(b, TensorLayout((n, k), (k, 1)), "global", (bn, k), (bid_y * bn, 0))
            ts_b = make_tensor("float16", layout_auto((bn, bk, k_pipe_max)), "shared")
            ts_a = make_tensor("float16", layout_auto((bm, bk, k_pipe_max)), "shared")
            tr_a = make_tensor("float16", layout_auto((bm, inst_k * 2)), "register")
            tr_c = make_tensor("float32", layout_auto((bm, bn)), "register")
            fill(tr_c, 0.0)

            txga = partition_src(tg_a, tiled_copy_a)
            txra = partition_dst(tr_a, auto_copy())
            txSa = partition_src(ts_a, auto_copy())
            txsa = partition_dst(ts_a, tiled_copy_a)
            txgb = partition_src(tg_b, tiled_copy_b)
            txsb = partition_dst(ts_b, tiled_copy_b)
            txSb = partition_B(ts_b, tiled_mma)

            syncthreads()

            k_blocks = cdiv(k, bk)
            k_tiles = cdiv(bk, inst_k)
            for s in range(k_pipe_max):
                copy(tiled_copy_a, txga[:, :, s], txsa[:, :, s], mbarrier=mbar_tma[s])
                copy(tiled_copy_b, txgb[:, :, s], txsb[:, :, s], mbarrier=mbar_tma[s])
                mbarrier_arrive(mbar_tma[s], tma_copy_tx)

            smem_pipe_read = 0
            smem_pipe_write = 0
            smem_pipe_release = 0
            read_phase = False
            release_phase = False
            wgmma_fence_operand(tr_c)
            for ko in range(k_pipe_mmas):
                mbarrier_wait(mbar_tma[smem_pipe_read], read_phase)
                copy(auto_copy(), txSa[:, :, 0, smem_pipe_read], txra[:, :, 0])
                for ki in range(k_tiles):
                    wgmma_fence()
                    mma(tiled_mma, tr_c, txra[:, :, ki % 2], txSb[:, :, ki, smem_pipe_read], tr_c)
                    if ki < k_tiles - 1:
                        copy(auto_copy(), txSa[:, :, ki + 1, smem_pipe_read], txra[:, :, (ki + 1) % 2])
                wgmma_commit_group()
                smem_pipe_read += 1
            wgmma_fence_operand(tr_c)

            for ko in range(k_blocks - k_pipe_mmas):
                mbarrier_wait(mbar_tma[smem_pipe_read], read_phase)

                wgmma_fence_operand(tr_c)
                copy(auto_copy(), txSa[:, :, 0, smem_pipe_read], txra[:, :, 0])
                for ki in range(k_tiles):
                    wgmma_fence()
                    mma(tiled_mma, tr_c, txra[:, :, ki % 2], txSb[:, :, ki, smem_pipe_read], tr_c)
                    if ki < k_tiles - 1:
                        copy(auto_copy(), txSa[:, :, ki + 1, smem_pipe_read], txra[:, :, (ki + 1) % 2])
                wgmma_commit_group()
                wgmma_wait_group(k_pipe_mmas)
                wgmma_fence_operand(tr_c)
                mbarrier_arrive(mbar_mma[smem_pipe_release])

                if ko + k_pipe_max < k_blocks:
                    mbarrier_wait(mbar_mma[smem_pipe_release], release_phase)
                    copy(
                        tiled_copy_a,
                        txga[:, :, ko + k_pipe_max],
                        txsa[:, :, smem_pipe_write],
                        mbarrier=mbar_tma[smem_pipe_write],
                    )
                    copy(
                        tiled_copy_b,
                        txgb[:, :, ko + k_pipe_max],
                        txsb[:, :, smem_pipe_write],
                        mbarrier=mbar_tma[smem_pipe_write],
                    )
                    mbarrier_arrive(mbar_tma[smem_pipe_write], tma_copy_tx)
                    smem_pipe_write += 1
                    if smem_pipe_write == k_pipe_max:
                        smem_pipe_write = 0

                smem_pipe_read += 1
                if smem_pipe_read == k_pipe_max:
                    smem_pipe_read = 0
                    read_phase = not read_phase
                smem_pipe_release += 1
                if smem_pipe_release == k_pipe_max:
                    smem_pipe_release = 0
                    release_phase = not release_phase

            wgmma_wait_group(0)
            wgmma_fence_operand(tr_c)

            tr_C = rearrange(cast(tr_c, f16), auto_layout, "register")

            tg_c = tensor_view(
                c[bid_x * bm : (bid_x + 1) * bm, bid_y * bn : (bid_y + 1) * bn],
                TensorLayout((bm, bn), (n, 1)),
                "global",
            )
            txgc = partition_src(tg_c, auto_copy())
            txrc = partition_dst(tr_C, auto_copy())
            copy(auto_copy((bm, bn)), txrc, txgc)

    func = script_module.build()
    return func


# Currently, we don't have copy instruction for non-power-of-2 tiles
# As a result, we cannot test instructions with non-power-of-2 tiles
# untill the TMA instruction is supported.
wgmma_ns = [8] + list(range(16, 257, 16))


@pytest.mark.requires_cuda_hopper
@pytest.mark.parametrize("m,n,k", [(1024, 1024, 1024)])
@pytest.mark.parametrize("wgmma_n", wgmma_ns)
def test_f8_hopper_gemm_multiple_stage_ss(m, n, k, wgmma_n, group_k=128):
    # hidet.option.cache_dir("./demo_f8_hopper_gemm")
    # hidet.option.search_space(2)
    # hidet.option.debug_cache_tuning()
    # hidet.option.save_lower_ir(True)
    # hidet.option.use_torch_stream(True)

    if wgmma_n != group_k:
        pytest.skip("skip because block size n is not equal to group_k")

    n = cdiv(n, wgmma_n) * wgmma_n
    func = f8_gemm_multiple_stage_ss(m, n, k, wgmma_n=wgmma_n, group_k=group_k)
    a, b, scale_a, scale_b, c = f8_quant_data(m, n, k, trans_b=True, return_hidet=True, group_k=group_k)

    def fn():
        func(a, b, c, scale_a, scale_b)

    mean = do_bench(fn, percentiles=None)
    print(f"{m}x{n}x{k} took {mean:.2f} ms, throughput: {2.0 * m * n * k / mean / 1e9:.2f} TFLOPS")
    func(a, b, c, scale_a, scale_b)


#    mean = do_bench(fn2, percentiles=None)
#    print(f"{m}x{n}x{k} took {mean:.2f} ms, throughput: {2.0 * m * n * k / mean / 1e9:.2f} TFLOPS")
#
#    c2 = a @ b.T
#
#    import numpy as np
#
#    np.set_printoptions(threshold=3000, linewidth=200, edgeitems=100)
#    np.testing.assert_allclose(actual=c.cpu().numpy(), desired=c2.cpu().numpy(), rtol=1e-2)


@pytest.mark.requires_cuda_hopper
@pytest.mark.parametrize("wgmma_n", wgmma_ns)
@pytest.mark.parametrize("m,n,k", [(1024, 1024, 1024)])
def test_hopper_gemm_multiple_stage_rs(m, n, k, wgmma_n):
    # hidet.option.cache_dir("./demo_hopper_gemm")
    # hidet.option.search_space(2)
    # hidet.option.debug_cache_tuning()
    # hidet.option.save_lower_ir(True)

    n = cdiv(n, wgmma_n) * wgmma_n
    func = gemm_multiple_stage_rs(m, n, k, wgmma_n=wgmma_n)
    a, b, c = data(m, n, k, trans_b=True, return_hidet=True)

    def fn():
        func(a, b, c)

    mean = do_bench(fn, percentiles=None)
    print(f"{m}x{n}x{k} took {mean:.2f} ms, throughput: {2.0 * m * n * k / mean / 1e9:.2f} TFLOPS")
    func(a, b, c)

    a = a.torch()
    b = b.torch()

    def fn2():
        c2 = a @ b.T

    mean = do_bench(fn2, percentiles=None)
    print(f"{m}x{n}x{k} took {mean:.2f} ms, throughput: {2.0 * m * n * k / mean / 1e9:.2f} TFLOPS")

    c2 = a @ b.T

    import numpy as np

    np.set_printoptions(threshold=3000, linewidth=200, edgeitems=100)
    np.testing.assert_allclose(actual=c.cpu().numpy(), desired=c2.cpu().numpy(), rtol=1e-2)


def gemm_single_stage_rs(m, n, k, wgmma_n=64):
    a = TensorLayout(((4, 8, 4), (2, 2, 2)), ((128, 1, 16), (64, 8, 512)))
    b = TensorLayout(((128,), (wgmma_n, 16)), ((0,), (1, wgmma_n)))
    c = TensorLayout(((4, 8, 4), (2, 2, wgmma_n // 8)), ((128, 1, 16), (64, 8, 512)))
    mma_atom = MmaAtom("warp_group", (64, wgmma_n, 16), a, b, c, c)
    wg_in_threadblock = Level("warp_group", "thread_block", (1, 1), TensorLayout((1, 1)), (2, 2))
    tiled_mma = TiledMma(mma_atom, [wg_in_threadblock])

    a_shape, a_tv = tiled_mma.a_tv_layout()
    b_shape, b_tv = tiled_mma.b_tv_layout()
    c_shape, c_tv = tiled_mma.c_tv_layout()
    from quant_utils import canonicalize

    a_t, a_v = canonicalize(a_tv)
    b_t, b_v = canonicalize(b_tv)
    c_t, c_v = canonicalize(c_tv)

    bm, inst_k = a_shape
    bn, inst_k_ = b_shape
    bm_, bn_ = c_shape
    assert bm == bm_ and bn == bn_ and inst_k == inst_k_
    threads = a_t.size()
    bk = 32
    tma_copy_tx = bn * bk * f16.nbytes

    copy_atom = CopyAtom("thread_block", (bn, bk), TensorLayout(((threads,), (bn, bk)), ((0,), (1, bn))))
    tiled_copy_b = TiledCopy(copy_atom, [])

    with hidet.script_module() as script_module:

        @hidet.script
        def func(a: f16[m, k], b_ptr: ~f16, c: f16[m, n]):
            attrs.func_kind = "cuda_kernel"
            attrs.cuda.block_dim = threads  # 8 warps
            attrs.cuda.grid_dim = cdiv(m, bm), cdiv(n, bn), 1
            attrs.cuda.dynamic_smem_bytes = 0

            bid_x = blockIdx.x
            bid_y = blockIdx.y

            mbar_tma = make_mbarriers(1)
            tg_a = tensor_view(a, TensorLayout((m, k), (k, 1)), "global", (bm, k), (bid_x * bm, 0))
            b = as_tensor_pointer(b_ptr, "float16", [n, k])
            tg_b = tensor_view(b, TensorLayout((n, k), (k, 1)), "global", (bn, k), (bid_y * bn, 0))
            ts_b = make_tensor("float16", layout_auto((bn, bk)), "shared")
            ts_a = make_tensor("float16", layout_auto((bm, bk)), "shared")
            tr_a = make_tensor("float16", layout_auto((bm, inst_k * 2)), "register")
            tr_c = make_tensor("float32", layout_auto((bm, bn)), "register")
            fill(tr_c, 0.0)
            syncthreads()

            txga = partition_src(tg_a, auto_copy())
            txra = partition_dst(tr_a, auto_copy())
            txSa = partition_src(ts_a, auto_copy())
            txsa = partition_dst(ts_a, auto_copy())
            txgb = partition_src(tg_b, tiled_copy_b)
            txsb = partition_dst(ts_b, tiled_copy_b)
            txSb = partition_B(ts_b, tiled_mma)

            k_tiles = cdiv(bk, inst_k)
            phase = False
            for ko in range(cdiv(k, bk)):
                copy(tiled_copy_b, txgb[:, :, ko], txsb[:, :], mbarrier=mbar_tma[0])
                mbarrier_arrive(mbar_tma[0], tma_copy_tx)
                copy(auto_copy((bm, bk)), txga[:, :, ko], txsa)
                # mbar_status = mbarrier_try_wait(mbar_tma[0], phase)
                cp_async_commit_group()
                cp_async_wait_group(allow_on_fly_groups=0)
                # if mbar_status:
                mbarrier_wait(mbar_tma[0], phase)
                phase = not phase
                syncthreads()

                copy(auto_copy(), txSa[:, :, 0], txra[:, :, 0])

                for ki in range(k_tiles):
                    if ki + 1 < k_tiles:
                        copy(auto_copy(), txSa[:, :, ki + 1], txra[:, :, (ki + 1) % 2])
                    mma(tiled_mma, tr_c, txra[:, :, ki % 2], txSb[:, :, ki], tr_c)
                wgmma_commit_group()
                wgmma_wait_group(0)
                syncthreads()

            tr_C = rearrange(cast(tr_c, f16), auto_layout, "register")

            tg_c = tensor_view(
                c[bid_x * bm : (bid_x + 1) * bm, bid_y * bn : (bid_y + 1) * bn],
                TensorLayout((bm, bn), (n, 1)),
                "global",
            )
            txgc = partition_src(tg_c, auto_copy())
            txrc = partition_dst(tr_C, auto_copy())
            mask_c = mask(auto_copy(), [m - bid_x * bm, n - bid_y * bn])
            copy(auto_copy((bm, bn)), txrc, txgc, mask_c)

    func = script_module.build()
    return func


@pytest.mark.requires_cuda_hopper
@pytest.mark.parametrize("wgmma_n", wgmma_ns)
@pytest.mark.parametrize("m, k", [(1024, 1024)])
def test_hopper_gemm_single_stage_rs(m, k, wgmma_n):
    # hidet.option.cache_dir("./demo_hopper_gemm")
    # hidet.option.search_space(2)
    # hidet.option.debug_cache_tuning()
    # hidet.option.save_lower_ir(True)
    if wgmma_n > 128:
        pytest.skip("skip because there is a bug need to be fixed")

    n = 1024
    func = gemm_single_stage_rs(m, n, k, wgmma_n=wgmma_n)
    a, b, c = data(m, n, k, trans_b=True, return_hidet=True)

    def fn():
        func(a, b, c)

    mean = do_bench(fn, percentiles=None)
    print(f"{m}x{n}x{k} took {mean:.2f} ms, throughput: {2.0 * m * n * k / mean / 1e9:.2f} TFLOPS")
    func(a, b, c)

    a = a.torch()
    b = b.torch()

    def fn2():
        c2 = a @ b.T

    mean = do_bench(fn2, percentiles=None)
    print(f"{m}x{n}x{k} took {mean:.2f} ms, throughput: {2.0 * m * n * k / mean / 1e9:.2f} TFLOPS")

    c2 = a @ b.T
    import numpy as np

    np.set_printoptions(threshold=3000, linewidth=200, edgeitems=100)
    np.testing.assert_allclose(actual=c.cpu().numpy(), desired=c2.cpu().numpy(), rtol=1e-2)


def gemm_multiple_stage_rs_auto(m, n, k, wgmma_n=64, trans_b=True):
    a = TensorLayout(((4, 8, 4), (2, 2, 2)), ((128, 1, 16), (64, 8, 512)))
    b = TensorLayout(((128,), (wgmma_n, 16)), ((0,), (1, wgmma_n)))
    c = TensorLayout(((4, 8, 4), (2, 2, wgmma_n // 8)), ((128, 1, 16), (64, 8, 512)))
    mma_atom = MmaAtom("warp_group", (64, wgmma_n, 16), a, b, c, c)
    wg_in_threadblock = Level("warp_group", "thread_block", (2, 1), TensorLayout((2, 1)), (1, 1))
    tiled_mma = TiledMma(mma_atom, [wg_in_threadblock])

    a_shape, a_tv = tiled_mma.a_tv_layout()
    b_shape, b_tv = tiled_mma.b_tv_layout()
    c_shape, c_tv = tiled_mma.c_tv_layout()
    from quant_utils import canonicalize

    a_t, _ = canonicalize(a_tv)
    _, _ = canonicalize(b_tv)
    _, _ = canonicalize(c_tv)

    bm, inst_k = a_shape
    bn, inst_k_ = b_shape
    bm_, bn_ = c_shape
    assert bm == bm_ and bn == bn_ and inst_k == inst_k_
    threads = a_t.size()
    num_producer_threads = 128
    num_consumer_threads = threads
    bk = 64
    tma_copy_tx = (bm * bk + bn * bk) * f16.nbytes
    k_pipe_max = 4

    with hidet.script_module() as script_module:

        @hidet.script
        def func(a: f16[m, k], b_ptr: ~f16, c: f16[m, n]):
            """
            H100 GEMM Kernel with Warp Specialization

            This kernel implements a high-performance GEMM (General Matrix Multiplication)
            optimized for NVIDIA H100 GPUs using warp specialization. The kernel splits
            computation into producer and consumer warps to overlap computation with data movement.

            Key Features:
            1. Warp Specialization: Dedicated producer warps for data movement and consumer warps for computation
            2. WGMMA Instructions: Uses Warp Group Matrix Multiply-Accumulate for efficient computation
            3. Double Buffering: Implements pipelined computation with multiple stages
            4. Memory Barriers: Efficient synchronization between producer and consumer warps
            5. Tensor Memory Access (TMA): Optimized global memory access patterns

            Args:
                a (f16[m, k]): Input matrix A in FP16 format
                b_ptr (~f16): Pointer to input matrix B in FP16 format
                c (f16[m, n]): Output matrix C in FP16 format
            """
            # Kernel Configuration
            attrs.func_kind = "cuda_kernel"
            attrs.cuda.block_dim = num_producer_threads + num_consumer_threads  # 12 warps total
            attrs.cuda.grid_dim = cdiv(m, bm), cdiv(n, bn), 1  # Grid dimensions based on matrix size
            attrs.cuda.min_blocks = 1
            attrs.cuda.dynamic_smem_bytes = 0  # No dynamic shared memory required

            # Block indices for grid-level parallelism
            bid_x = blockIdx.x
            bid_y = blockIdx.y

            # Initialize memory barriers for producer-consumer synchronization
            mbar_tma = make_mbarriers(k_pipe_max)  # For TMA operations
            mbar_mma = make_mbarriers(k_pipe_max)  # For MMA operations

            # Set up tensor views for global memory access
            # Matrix A: Global memory view with strided layout
            tg_a = tensor_view(a, TensorLayout((m, k), (k, 1)), "global", (bm, k), (bid_x * bm, 0))
            # Matrix B: Global memory view with strided layout
            b = as_tensor_pointer(b_ptr, "float16", [n, k])
            tg_b = tensor_view(b, TensorLayout((n, k), (k, 1)), "global", (bn, k), (bid_y * bn, 0))

            # Allocate shared memory tensors with pipelined layout
            ts_b = make_tensor("float16", layout_auto((bn, bk, k_pipe_max)), "shared")
            ts_a = make_tensor("float16", layout_auto((bm, bk, k_pipe_max)), "shared")

            # Initial synchronization to ensure shared memory is ready
            syncthreads()

            # Producer Warp Group: Responsible for data movement from global to shared memory
            with warp_groups_producer([0], num_regs=40):
                # Pipeline control variables
                smem_pipe_write = 0
                write_phase = True

                # Set up tensor partitions for efficient data movement
                txga = partition_src(tg_a, auto_copy())
                txsa = partition_dst(ts_a, auto_copy())
                txgb = partition_src(tg_b, auto_copy())
                txsb = partition_dst(ts_b, auto_copy())

                # Main producer loop: Move data from global to shared memory
                k_blocks = cdiv(k, bk)
                for ko in range(k_blocks):
                    # Wait for previous MMA operations to complete if pipeline is full
                    if ko >= k_pipe_max:
                        mbarrier_wait(mbar_mma[smem_pipe_write], write_phase)

                    # Copy data from global to shared memory using TMA
                    copy(
                        auto_copy((bm, bk)),
                        txga[:, :, ko],
                        txsa[:, :, smem_pipe_write],
                        mbarrier=mbar_tma[smem_pipe_write],
                    )
                    copy(
                        auto_copy((bn, bk)),
                        txgb[:, :, ko],
                        txsb[:, :, smem_pipe_write],
                        mbarrier=mbar_tma[smem_pipe_write],
                    )

                    # Signal completion of TMA operation
                    mbarrier_arrive(mbar_tma[smem_pipe_write], tma_copy_tx)

                    # Update pipeline stage
                    smem_pipe_write += 1
                    if smem_pipe_write == k_pipe_max:
                        smem_pipe_write = 0
                        write_phase = not write_phase

            # Consumer Warp Group: Responsible for matrix multiplication computation
            with warp_groups_consumer([1, 2], num_regs=224):
                # Pipeline control variables
                smem_pipe_read = 0
                read_phase = False
                smem_pipe_release = 0
                release_phase = False

                # Allocate register tensors for computation
                tr_a = make_tensor("float16", layout_auto((bm, inst_k * 2)), "register")
                tr_c = make_tensor("float32", layout_auto((bm, bn)), "register")
                fill(tr_c, 0.0)  # Initialize accumulation register

                # Set up tensor partitions for computation
                txra = partition_dst(tr_a, auto_copy())
                txSa = partition_src(ts_a, auto_copy())
                txSb = partition_B(ts_b, tiled_mma)

                # Main computation loop
                k_blocks = cdiv(k, bk)
                k_tiles = cdiv(bk, inst_k)

                # Initialize WGMMA pipeline
                wgmma_fence_operand(tr_c)
                mbarrier_wait(mbar_tma[smem_pipe_read], read_phase)
                copy(auto_copy(), txSa[:, :, 0, smem_pipe_read], txra[:, :, 0])

                # Main computation loop with pipelined WGMMA operations
                for ki in range(k_tiles - 1):
                    # Double buffering: Load next tile while computing current tile
                    copy(auto_copy(), txSa[:, :, ki + 1, smem_pipe_read], txra[:, :, (ki + 1) % 2])
                    wgmma_fence()

                    # Perform matrix multiplication using WGMMA
                    mma(tiled_mma, tr_c, txra[:, :, ki % 2], txSb[:, :, ki, smem_pipe_read], tr_c)
                    wgmma_commit_group()

                # Handle last tile of first iteration
                read_stage = smem_pipe_read
                smem_pipe_read += 1
                if smem_pipe_read == k_pipe_max:
                    smem_pipe_read = 0
                    read_phase = not read_phase

                # Complete first iteration
                wgmma_wait_group(2)
                wgmma_fence()
                mma(tiled_mma, tr_c, txra[:, :, (k_tiles - 1) % 2], txSb[:, :, k_tiles - 1, read_stage], tr_c)
                wgmma_commit_group()

                # Main pipeline loop
                mbarrier_wait(mbar_tma[smem_pipe_read], read_phase)
                copy(auto_copy(), txSa[:, :, 0, smem_pipe_read], txra[:, :, 0])
                wgmma_wait_group(2)
                wgmma_fence_operand(tr_c)

                # Process remaining blocks with pipelined computation
                for ko in range(k_blocks - 2):
                    read_stage = smem_pipe_read
                    smem_pipe_read += 1
                    if smem_pipe_read == k_pipe_max:
                        smem_pipe_read = 0
                        read_phase = not read_phase

                    wgmma_fence_operand(tr_c)
                    for ki in range(k_tiles):
                        # Load next tile while computing current tile
                        if ki == k_tiles - 1:
                            mbarrier_wait(mbar_tma[smem_pipe_read], read_phase)
                            copy(auto_copy(), txSa[:, :, 0, smem_pipe_read], txra[:, :, 0])
                        else:
                            copy(auto_copy(), txSa[:, :, ki + 1, read_stage], txra[:, :, (ki + 1) % 2])

                        # Perform matrix multiplication
                        wgmma_fence()
                        mma(tiled_mma, tr_c, txra[:, :, ki % 2], txSb[:, :, ki, read_stage], tr_c)
                        wgmma_commit_group()
                        wgmma_wait_group(2)

                        # Release shared memory for producer
                        if ki == 1:
                            mbarrier_arrive(mbar_mma[smem_pipe_release])
                            smem_pipe_release += 1
                            if smem_pipe_release == k_pipe_max:
                                smem_pipe_release = 0
                                release_phase = not release_phase
                    wgmma_fence_operand(tr_c)
                wgmma_fence_operand(tr_c)

                # Final computation phase
                wgmma_fence_operand(tr_c)
                for ki in range(k_tiles - 1):
                    copy(auto_copy(), txSa[:, :, ki + 1, smem_pipe_read], txra[:, :, (ki + 1) % 2])
                    wgmma_fence()
                    mma(tiled_mma, tr_c, txra[:, :, ki % 2], txSb[:, :, ki, smem_pipe_read], tr_c)
                    wgmma_commit_group()
                    wgmma_wait_group(2)
                    if ki == 1:
                        mbarrier_arrive(mbar_mma[smem_pipe_release])
                        smem_pipe_release += 1
                        if smem_pipe_release == k_pipe_max:
                            smem_pipe_release = 0
                            release_phase = not release_phase

                # Complete final computation
                wgmma_fence()
                mma(tiled_mma, tr_c, txra[:, :, (k_tiles - 1) % 2], txSb[:, :, k_tiles - 1, smem_pipe_read], tr_c)
                wgmma_fence_operand(tr_c)

                # Final synchronization
                wgmma_wait_group(0)
                mbarrier_arrive(mbar_mma[smem_pipe_release])

                # Convert result to FP16 and prepare for global memory write
                tr_C = rearrange(cast(tr_c, f16), auto_layout, "register")

                # Write result back to global memory
                tg_c = tensor_view(
                    c[bid_x * bm : (bid_x + 1) * bm, bid_y * bn : (bid_y + 1) * bn],
                    TensorLayout((bm, bn), (n, 1)),
                    "global",
                )
                txgc = partition_src(tg_c, auto_copy())
                txrc = partition_dst(tr_C, auto_copy())
                mask_c = mask(auto_copy(), [m - bid_x * bm, n - bid_y * bn])
                copy(auto_copy((bm, bn)), txrc, txgc, mask_c)

    func = script_module.build()
    return func


@pytest.mark.requires_cuda_hopper
@pytest.mark.parametrize("wgmma_n", wgmma_ns)
@pytest.mark.parametrize("m,n,k", [(1024, 1024, 1024)])
def test_hopper_gemm_multiple_stage_rs_auto(m, n, k, wgmma_n):
    # hidet.option.cache_dir("./demo_hopper_gemm")
    # hidet.option.search_space(2)
    # hidet.option.debug_cache_tuning()
    # hidet.option.save_lower_ir(True)

    func = gemm_multiple_stage_rs_auto(m, n, k, wgmma_n=wgmma_n)
    a, b, c = data(m, n, k, trans_b=True, return_hidet=True)

    def fn():
        func(a, b, c)

    mean = do_bench(fn, percentiles=None)
    print(f"{m}x{n}x{k} took {mean:.2f} ms, throughput: {2.0 * m * n * k / mean / 1e9:.2f} TFLOPS")
    func(a, b, c)

    a = a.torch()
    b = b.torch()

    def fn2():
        c2 = a @ b.T

    mean = do_bench(fn2, percentiles=None)
    print(f"{m}x{n}x{k} took {mean:.2f} ms, throughput: {2.0 * m * n * k / mean / 1e9:.2f} TFLOPS")

    c2 = a @ b.T

    import numpy as np

    np.set_printoptions(threshold=3000, linewidth=200, edgeitems=100)
    np.testing.assert_allclose(actual=c.cpu().numpy(), desired=c2.cpu().numpy(), rtol=1e-2)


if __name__ == "__main__":
    # test_hopper_gemm_single_stage_rs(4096, 4096, 16)
    #    test_hopper_gemm_multiple_stage_rs_auto(4096, 4096, 4096, 256)
    #    test_hopper_gemm_multiple_stage_rs_warp_specialized(1024, 1024, 1024, 256)
    #    test_hopper_gemm_multiple_stage_rs_warp_specialized(2048, 2048, 2048, 256)
    test_hopper_gemm_multiple_stage_rs(1024, 1024, 1024, 256)
    # test_hopper_gemm_multiple_stage_rs_auto(2048, 22016, 2048, 256)
    #    test_hopper_gemm_single_stage_rs(1024, 1024, 192)
    # test_hopper_gemm_multiple_stage_rs_warp_specialized(4096 + 8, 4096 - 8, 4096, 256)
    #    test_hopper_gemm_multiple_stage_rs_warp_specialized(256, 14336, 4096, 256)
    #    test_hopper_gemm_multiple_stage_rs_warp_specialized(512, 8192, 512, 256)
    #    test_hopper_gemm_multiple_stage_rs_warp_specialized(4096, 28672, 8192, 256)
    #
    # test_hopper_gemm_multiple_stage_rs_auto(1024, 1024, 1024, 64)
    # test_hopper_gemm_multiple_stage_rs_auto(1024, 1024, 1024, 48)
    # test_hopper_gemm_multiple_stage_rs_auto(1024, 1024, 1024, 160)
    # test_hopper_gemm_multiple_stage_rs_auto(1024, 1024, 1024, 96)
    # test_hopper_gemm_multiple_stage_rs_auto(1024, 1024, 1024, 224)

    # test_hopper_gemm_multiple_stage_rs_auto(1024, 1024, 1024, 64)
    # test_hopper_gemm_multiple_stage_rs_auto(1024, 1024, 1024, 128)
    # test_hopper_gemm_multiple_stage_rs_auto(1024, 1024, 1024, 256)
    # test_hopper_gemm_multiple_stage_rs(1024, 1024, 1024, 256)
    # test_hopper_gemm_multiple_stage_rs(2048, 2048, 2048, 256)
    # test_hopper_gemm_multiple_stage_rs(4096, 4096, 4096, 256)
    # test_hopper_gemm_multiple_stage_rs(256, 14336, 4096, 256)
    # test_hopper_gemm_multiple_stage_rs(512, 8192, 512, 256)
    # test_hopper_gemm_multiple_stage_rs(4096, 28672, 8192, 256)
    # test_f8_hopper_gemm_multiple_stage_ss(4096 + 8, 4096, 7168, 128)
