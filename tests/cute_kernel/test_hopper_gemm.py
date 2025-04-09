import torch
import pytest

from hidet.lang.types import f16, f32
from hidet.lang import attrs
from hidet.lang.cuda import blockIdx, threadIdx
from hidet.ir.primitives.cuda.wgmma import wgmma_fence, wgmma_commit_group, wgmma_wait_group
from hidet.lang.cuda import syncthreads, cp_async_commit_group, cp_async_wait_group
from hidet.lang.constructs.declare import as_tensor_pointer
from hidet.utils.py import cdiv

import hidet
from hidet.ir.cute.algorithm import MmaAtom, TiledMma, auto_copy
from hidet.ir.cute.layout import TensorLayout, Level
from hidet.ir.cute import layout_auto, auto_layout

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
)


def data(M, N, K, trans_a=False, trans_b=False, dtype="float16", device="cuda", return_hidet=False):
    dtype = getattr(torch, dtype)
    lo = -2
    hi = 2
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


def gemm_rs(m, n, k, wgmma_n=64, trans_b=True):
    a = TensorLayout(((4, 8, 4), (2, 2, 2)), ((128, 1, 16), (64, 8, 512)))
    b = TensorLayout(((128,), (wgmma_n, 16)), ((0,), (1, wgmma_n)))
    c = TensorLayout(((4, 8, 4), (2, 2, wgmma_n // 8)), ((128, 1, 16), (64, 8, 512)))
    mma_atom = MmaAtom("warp_group", (64, wgmma_n, 16), a, b, c, c)
    wg_in_threadblock = Level("warp_group", "thread_block", (2, 2), TensorLayout((2, 2)))
    tiled_mma = TiledMma(mma_atom, [wg_in_threadblock])

    a_shape, a_tv = tiled_mma.a_tv_layout()
    b_shape, b_tv = tiled_mma.b_tv_layout()
    c_shape, c_tv = tiled_mma.c_tv_layout()
    from quant_utils import canonicalize

    a_t, a_v = canonicalize(a_tv)
    b_t, b_v = canonicalize(b_tv)
    c_t, c_v = canonicalize(c_tv)

    bm, bk = a_shape
    bn, bk_ = b_shape
    bm_, bn_ = c_shape
    assert bm == bm_ and bn == bn_ and bk == bk_
    print(bm, bn, bk)
    threads = a_t.size()

    with hidet.script_module() as script_module:

        @hidet.script
        def func(a: f16[m, k], b_ptr: ~f16, c: f16[m, n]):
            attrs.func_kind = "cuda_kernel"
            attrs.cuda.block_dim = threads  # 8 warps
            attrs.cuda.grid_dim = cdiv(m, bm), cdiv(n, bn), 1
            attrs.cuda.dynamic_smem_bytes = 0

            bid_x = blockIdx.x
            bid_y = blockIdx.y

            tg_a = tensor_view(a[bid_x * bm : (bid_x + 1) * bm, :], TensorLayout((bm, k), (k, 1)), "global")
            if trans_b:
                b = as_tensor_pointer(b_ptr, "float16", [n, k])
                tg_b = tensor_view(b[bid_y * bn : (bid_y + 1) * bn, :], TensorLayout((bn, k), (k, 1)), "global")
            else:
                b = as_tensor_pointer(b_ptr, "float16", [k, n])
                tg_b = tensor_view(b[:, bid_y * bn : (bid_y + 1) * bn], TensorLayout((bn, k), (1, n)), "global")
            ts_b = make_tensor("float16", layout_auto((bn, bk)), "shared")
            tr_a = make_tensor("float16", layout_auto((bm, bk)), "register")
            tr_c = make_tensor("float32", layout_auto((bm, bn)), "register")
            fill(tr_c, 0.0)

            txga = partition_src(tg_a, auto_copy())
            txra = partition_dst(tr_a, auto_copy())
            txgb = partition_src(tg_b, auto_copy())
            txsb = partition_dst(ts_b, auto_copy())
            txSb = partition_B(ts_b, tiled_mma)

            for ki in range(cdiv(k, bk)):
                copy(auto_copy((bm, bk)), txga[:, :, ki], txra)
                copy(auto_copy((bn, bk)), txgb[:, :, ki], txsb)
                cp_async_commit_group()
                cp_async_wait_group(allow_on_fly_groups=0)
                syncthreads()
                wgmma_fence()
                mma(tiled_mma, tr_c, txra, txSb, tr_c)
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
            copy(auto_copy((bm, bn)), txrc, txgc)

    func = script_module.build()
    return func


def gemm_ss(m, n, k, wgmma_n=64, trans_b=True):
    a = TensorLayout(((4, 8, 4), (2, 2, 2)), ((128, 1, 16), (64, 8, 512)))
    b = TensorLayout(((128,), (wgmma_n, 16)), ((0,), (1, wgmma_n)))
    c = TensorLayout(((4, 8, 4), (2, 2, wgmma_n // 8)), ((128, 1, 16), (64, 8, 512)))
    mma_atom = MmaAtom("warp_group", (64, wgmma_n, 16), a, b, c, c)
    wg_in_threadblock = Level("warp_group", "thread_block", (2, 2), TensorLayout((2, 2)))
    tiled_mma = TiledMma(mma_atom, [wg_in_threadblock])

    a_shape, a_tv = tiled_mma.a_tv_layout()
    b_shape, b_tv = tiled_mma.b_tv_layout()
    c_shape, c_tv = tiled_mma.c_tv_layout()
    from quant_utils import canonicalize

    a_t, a_v = canonicalize(a_tv)
    b_t, b_v = canonicalize(b_tv)
    c_t, c_v = canonicalize(c_tv)
    print(a_t, a_v)
    print(b_t, b_v)
    print(c_t, c_v)

    bm, bk = a_shape
    bn, bk_ = b_shape
    bm_, bn_ = c_shape
    assert bm == bm_ and bn == bn_ and bk == bk_
    print(bm, bn, bk)
    threads = a_t.size()

    with hidet.script_module() as script_module:

        @hidet.script
        def func(a: f16[m, k], b_ptr: ~f16, c: f16[m, n]):
            attrs.func_kind = "cuda_kernel"
            attrs.cuda.block_dim = threads  # 8 warps
            attrs.cuda.grid_dim = cdiv(m, bm), cdiv(n, bn), 1
            attrs.cuda.dynamic_smem_bytes = 0

            bid_x = blockIdx.x
            bid_y = blockIdx.y

            tg_a = tensor_view(a[bid_x * bm : (bid_x + 1) * bm, :], TensorLayout((bm, k), (k, 1)), "global")
            if trans_b:
                b = as_tensor_pointer(b_ptr, "float16", [n, k])
                tg_b = tensor_view(b[bid_y * bn : (bid_y + 1) * bn, :], TensorLayout((bn, k), (k, 1)), "global")
            else:
                b = as_tensor_pointer(b_ptr, "float16", [k, n])
                tg_b = tensor_view(b[:, bid_y * bn : (bid_y + 1) * bn], TensorLayout((bn, k), (1, n)), "global")
            ts_b = make_tensor("float16", layout_auto((bn, bk)), "shared")
            ts_a = make_tensor("float16", layout_auto((bm, bk)), "register")
            tr_c = make_tensor("float32", layout_auto((bm, bn)), "register")
            fill(tr_c, 0.0)

            txga = partition_src(tg_a, auto_copy())
            txsa = partition_dst(ts_a, auto_copy())
            txgb = partition_src(tg_b, auto_copy())
            txsb = partition_dst(ts_b, auto_copy())
            txSa = partition_A(ts_a, tiled_mma)
            txSb = partition_B(ts_b, tiled_mma)

            for ki in range(cdiv(k, bk)):
                copy(auto_copy((bm, bk)), txga[:, :, ki], txsa)
                copy(auto_copy((bn, bk)), txgb[:, :, ki], txsb)
                cp_async_commit_group()
                cp_async_wait_group(allow_on_fly_groups=0)
                syncthreads()
                wgmma_fence()
                mma(tiled_mma, tr_c, txSa, txSb, tr_c)
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
            copy(auto_copy((bm, bn)), txrc, txgc)

    func = script_module.build()
    return func


def gemm_single_stage_rs(m, n, k, wgmma_n=64, trans_b=True):
    a = TensorLayout(((4, 8, 4), (2, 2, 2)), ((128, 1, 16), (64, 8, 512)))
    b = TensorLayout(((128,), (wgmma_n, 16)), ((0,), (1, wgmma_n)))
    c = TensorLayout(((4, 8, 4), (2, 2, wgmma_n // 8)), ((128, 1, 16), (64, 8, 512)))
    mma_atom = MmaAtom("warp_group", (64, wgmma_n, 16), a, b, c, c)
    wg_in_threadblock = Level("warp_group", "thread_block", (2, 2), TensorLayout((2, 2)))
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
    bk = 64

    with hidet.script_module() as script_module:

        @hidet.script
        def func(a: f16[m, k], b_ptr: ~f16, c: f16[m, n]):
            attrs.func_kind = "cuda_kernel"
            attrs.cuda.block_dim = threads  # 8 warps
            attrs.cuda.grid_dim = cdiv(m, bm), cdiv(n, bn), 1
            attrs.cuda.dynamic_smem_bytes = 0

            bid_x = blockIdx.x
            bid_y = blockIdx.y

            tg_a = tensor_view(a[bid_x * bm : (bid_x + 1) * bm, :], TensorLayout((bm, k), (k, 1)), "global")
            if trans_b:
                b = as_tensor_pointer(b_ptr, "float16", [n, k])
                tg_b = tensor_view(b[bid_y * bn : (bid_y + 1) * bn, :], TensorLayout((bn, k), (k, 1)), "global")
            else:
                b = as_tensor_pointer(b_ptr, "float16", [k, n])
                tg_b = tensor_view(b[:, bid_y * bn : (bid_y + 1) * bn], TensorLayout((bn, k), (1, n)), "global")
            ts_b = make_tensor("float16", layout_auto((bn, bk)), "shared")
            ts_a = make_tensor("float16", layout_auto((bm, bk)), "shared")
            tr_a = make_tensor("float16", layout_auto((bm, inst_k * 2)), "register")
            tr_c = make_tensor("float32", layout_auto((bm, bn)), "register")
            fill(tr_c, 0.0)

            txga = partition_src(tg_a, auto_copy())
            txra = partition_dst(tr_a, auto_copy())
            txSa = partition_src(ts_a, auto_copy())
            txsa = partition_dst(ts_a, auto_copy())
            txgb = partition_src(tg_b, auto_copy())
            txsb = partition_dst(ts_b, auto_copy())
            txSb = partition_B(ts_b, tiled_mma)

            k_tiles = cdiv(bk, inst_k)
            for ko in range(cdiv(k, bk)):
                copy(auto_copy((bm, bk)), txga[:, :, ko], txsa)
                copy(auto_copy((bn, bk)), txgb[:, :, ko], txsb[:, :])
                cp_async_commit_group()
                cp_async_wait_group(allow_on_fly_groups=0)
                syncthreads()

                copy(auto_copy(), txSa[:, :, 0], txra[:, :, 0])

                for ki in range(k_tiles):
                    wgmma_fence()
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
            copy(auto_copy((bm, bn)), txrc, txgc)

    func = script_module.build()
    return func


# Currently, we don't have copy instruction for non-power-of-2 tiles
# As a result, we cannot test instructions with non-power-of-2 tiles
# untill the TMA instruction is supported.
wgmma_ns = [8, 16, 32, 64, 128, 192]


@pytest.mark.requires_cuda_hopper
@pytest.mark.parametrize("trans_b", [True, False])
@pytest.mark.parametrize("wgmma_n", wgmma_ns)
@pytest.mark.parametrize("m, k", [(1024, 1024)])
def test_hopper_gemm_rs(m, k, wgmma_n, trans_b):
    # hidet.option.cache_dir("./demo_wgmma")
    # hidet.option.search_space(2)
    # hidet.option.debug_cache_tuning()
    # hidet.option.save_lower_ir(True)

    if not trans_b and wgmma_n == 192:
        return

    n = max(1024, wgmma_n * 16)
    func = gemm_rs(m, n, k, wgmma_n=wgmma_n, trans_b=trans_b)
    a, b, c = data(m, n, k, trans_b=trans_b, return_hidet=True)
    func(a, b, c)

    a = a.torch()
    b = b.torch()
    if trans_b:
        c2 = a @ b.T
    else:
        c2 = a @ b
    import numpy as np

    np.set_printoptions(threshold=3000, linewidth=200, edgeitems=100)
    np.testing.assert_allclose(actual=c.cpu().numpy(), desired=c2.cpu().numpy(), rtol=1e-2)


@pytest.mark.requires_cuda_hopper
@pytest.mark.parametrize("trans_b", [True, False])
@pytest.mark.parametrize("wgmma_n", wgmma_ns)
@pytest.mark.parametrize("m, k", [(1024, 1024)])
def test_hopper_gemm_ss(m, k, wgmma_n, trans_b):
    # hidet.option.cache_dir("./demo_wgmma")
    # hidet.option.search_space(2)
    # hidet.option.debug_cache_tuning()
    # hidet.option.save_lower_ir(True)

    if not trans_b and wgmma_n == 192:
        return

    n = max(1024, wgmma_n * 16)
    func = gemm_ss(m, n, k, wgmma_n=wgmma_n, trans_b=trans_b)
    a, b, c = data(m, n, k, trans_b=trans_b, return_hidet=True)
    func(a, b, c)

    a = a.torch()
    b = b.torch()
    if trans_b:
        c2 = a @ b.T
    else:
        c2 = a @ b
    import numpy as np

    np.set_printoptions(threshold=3000, linewidth=200, edgeitems=100)
    np.testing.assert_allclose(actual=c.cpu().numpy(), desired=c2.cpu().numpy(), rtol=1e-2)


@pytest.mark.requires_cuda_hopper
@pytest.mark.parametrize("trans_b", [True, False])
@pytest.mark.parametrize("wgmma_n", wgmma_ns)
@pytest.mark.parametrize("m, k", [(1024, 1024)])
def test_hopper_gemm_single_stage_rs(m, k, wgmma_n, trans_b):
    # hidet.option.cache_dir("./demo_wgmma")
    # hidet.option.search_space(2)
    # hidet.option.debug_cache_tuning()
    # hidet.option.save_lower_ir(True)

    if not trans_b and wgmma_n == 192:
        return

    print(trans_b, wgmma_n)

    n = max(1024, wgmma_n * 16)
    func = gemm_single_stage_rs(m, n, k, wgmma_n=wgmma_n, trans_b=trans_b)
    a, b, c = data(m, n, k, trans_b=trans_b, return_hidet=True)
    func(a, b, c)

    a = a.torch()
    b = b.torch()
    if trans_b:
        c2 = a @ b.T
    else:
        c2 = a @ b
    import numpy as np

    np.set_printoptions(threshold=3000, linewidth=200, edgeitems=100)
    np.testing.assert_allclose(actual=c.cpu().numpy(), desired=c2.cpu().numpy(), rtol=1e-2)


if __name__ == "__main__":
    test_hopper_gemm_single_stage_rs(1024, 1024, 128, True)
    # test_hopper_gemm_rs(128, 32, 128, False)
