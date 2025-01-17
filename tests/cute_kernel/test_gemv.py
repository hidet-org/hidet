import hidet
import pytest
import torch
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
    cast,
    exp,
    reduce_sum,
    reduce_max,
    partition_A,
    elementwise_max,
    broadcast_to,
    fill,
    cute_atomic_add,
)
from hidet.lang.mapping import auto_map
from hidet.ir.primitives.cuda.mutex import release_seq_semaphore, acquire_seq_semaphore

from quant_utils import canonicalize, bench


def gemv(n, k, k_parallel_parts):
    from hidet.lang.types import u32, i32, f16, f32
    from hidet.lang import attrs
    from hidet.lang import shared_tensor, register_tensor
    from hidet.lang.cuda import syncthreads, cp_async_wait_all
    from hidet.lang.cuda import blockIdx, threadIdx, dynamic_shared_memory
    from hidet.lang.cuda import cp_async, cp_async_commit_group, cp_async_wait_group

    from hidet.ir.cute.algorithm import auto_copy, auto_mma
    from hidet.ir.cute import auto_layout, layout_auto

    from hidet.utils.py import cdiv
    from hidet.lang import grid

    bk = 128
    bn = 128
    threads = 128
    tn = cdiv(n, bn)
    k_partition = bk
    while k_partition * k_parallel_parts < k:
        k_partition += bk

    with hidet.script_module() as script_module:

        @hidet.script
        def func(x: f16[k], w1: f16[k, n], y: f16[k_parallel_parts, n]):
            attrs.func_kind = "cuda_kernel"
            attrs.cuda.block_dim = threads
            attrs.cuda.grid_dim = k_parallel_parts, tn
            attrs.cuda.dynamic_smem_bytes = 0

            k_part = blockIdx.x
            pid_n = blockIdx.y

            tg_w1 = tensor_view(w1[k_part * k_partition :, pid_n * bn :], TensorLayout((bn, k), (1, n)), "global")
            tg_x = tensor_view(x[k_part * k_partition :], TensorLayout((bn, k), (0, 1)), "global")
            txgx = partition_src(tg_x, auto_copy())
            txgw1 = partition_src(tg_w1, auto_copy())
            tr_y_total = make_tensor("float32", layout_auto((bn, bk)), "register")
            fill(tr_y_total, 0.0)

            ko = (k_partition + bk - 1) // bk
            for j in range(ko):
                tr_w1 = make_tensor("float16", layout_auto((bn, bk)), "register")
                tr_x = make_tensor("float16", layout_auto((bn, bk), (0, 1)), "register")
                txrx = partition_dst(tr_x, auto_copy())
                txrw1 = partition_dst(tr_w1, auto_copy())
                copy(auto_copy((bk, bn)), txgx[:, :, j], txrx)
                copy(auto_copy((bk, bn)), txgw1[:, :, j], txrw1)

                tr_y = cast(tr_x, f32) * cast(tr_w1, f32)
                tr_y_total = tr_y_total + tr_y

            tr_y_f16 = cast(reduce_sum(tr_y_total, 1), f16)
            tg_y = tensor_view(y[k_part, pid_n * bn :], TensorLayout((bn, bk), (1, 0)), "global")
            txgy = partition_dst(tg_y, auto_copy())
            txry_f16 = partition_src(tr_y_f16, auto_copy())
            copy(auto_copy((bn, bk)), txry_f16, txgy)

    func = script_module.build()
    # a_mem = hidet.empty([m, k], device="cuda")
    # b_mem = hidet.empty([k, n], device="cuda")
    # c_mem = hidet.empty([m, n], device="cuda")
    # func(a_mem, b_mem, c_mem)
    return func


def data(k, n, k_parallel_parts, dtype="float16", device="cuda", return_hidet=False):
    dtype = getattr(torch, dtype)
    lo = -3
    hi = 3
    x = torch.randint(low=lo, high=hi, size=(k,), dtype=dtype, device=device)
    w1 = torch.randint(low=lo, high=hi, size=(k, n), dtype=dtype, device=device)
    y = torch.empty((n,), dtype=dtype, device=device)
    y_parts = torch.empty((k_parallel_parts, n), dtype=dtype, device=device)

    if return_hidet:
        x = hidet.from_torch(x)
        w1 = hidet.from_torch(w1)
        y = hidet.from_torch(y)
        y_parts = hidet.from_torch(y_parts)

    return x, w1, y, y_parts


@pytest.mark.requires_cuda
@pytest.mark.parametrize("n,k", [(4096, 4096 * 4), (4096 * 4, 4096)])
def test_gemv(n, k):
    # hidet.option.cache_dir("./demo_gemv")
    # hidet.option.search_space(2)
    # hidet.option.debug_cache_tuning()
    # hidet.option.save_lower_ir(True)

    k_parallel_parts = 4
    func = gemv(n, k, k_parallel_parts)
    x, w1, y, y_parts = data(k, n, k_parallel_parts)
    bk = 128
    blocks = (k + bk - 1) // bk
    mean, min_lat, max_lat = bench(func, (x, w1, y_parts))
    from hidet.ir.dtypes import f16

    memory = f16.nbytes * (k + k * n + n)
    print("time={:.3f} ms, bandwidth={:.3f} GB/s".format(mean, memory / (1e6 * mean)))

    def fn():
        func(x, w1, y_parts)
        return y_parts.sum(0)

    from hidet.utils.benchmark import do_bench

    mean = do_bench(fn, percentiles=None)
    print("time={:.3f} ms, bandwidth={:.3f} GB/s".format(mean, memory / (1e6 * mean)))
    y = fn()
    print(y_parts)
    print(y)
    print(w1.shape)

    def fn():
        return x @ w1

    from hidet.utils.benchmark import do_bench

    mean, min_lat, max_lat = bench(fn, ())
    mean = do_bench(fn, percentiles=None)
    print("time={:.3f} ms, bandwidth={:.3f} GB/s".format(mean, memory / (1e6 * mean)))

    y2 = fn()
    import numpy as np

    print(y)
    print(y2)
    np.set_printoptions(threshold=3000, linewidth=200, edgeitems=100)
    np.testing.assert_allclose(actual=y.cpu().numpy(), desired=y2.cpu().numpy(), rtol=1e-2)
