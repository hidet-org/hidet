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
)
from hidet.lang.mapping import auto_map
import torch


def weight_quantization_4bits_trans(m, n):
    from hidet.lang.types import u32, i32, f16, f32, i4, u4
    from hidet.lang import attrs
    from hidet.lang import shared_tensor, register_tensor
    from hidet.lang.cuda import syncthreads, cp_async_wait_all
    from hidet.lang.cuda import blockIdx, threadIdx, dynamic_shared_memory
    from hidet.lang.cuda import cp_async, cp_async_commit_group, cp_async_wait_group

    from hidet.ir.cute.algorithm import auto_copy, auto_mma
    from hidet.ir.cute import auto_layout

    from hidet.utils.py import cdiv
    from hidet.lang import grid
    from hidet.ir.expr import Expr, cast

    from hidet.ir.primitives import cvtv
    from hidet.ir.primitives.cuda.cvt import cvtv_func

    bm = 32
    bn = 128
    threads = 128

    def cvt_f16x8_to_u4x8(x: Expr):
        return u4(x)

    with hidet.script_module() as script_module:

        @hidet.script
        def func(w: f16[m, n], wq: u4[n, m]):
            attrs.func_kind = "cuda_kernel"
            attrs.cuda.block_dim = threads
            attrs.cuda.grid_dim = cdiv(m, bm) * cdiv(n, bn), 1
            attrs.cuda.dynamic_smem_bytes = 0

            pid = blockIdx.x
            num_pid_n = cdiv(n, bn)
            pid_m = pid // num_pid_n
            pid_n = pid % num_pid_n

            tg_w = tensor_view(w[pid_m * bm :, pid_n * bn :], TensorLayout((bm, bn), (n, 1)), "global")
            tr_w = make_tensor("float16", auto_layout, "register")

            txgx_w = partition_src(tg_w, auto_copy())
            txrx_w = partition_dst(tr_w, auto_copy())
            copy(auto_copy((bm, bn)), txgx_w, txrx_w)

            tr_w_cvt = rearrange(tr_w, auto_layout, "register")

            tr_wq = arithmetic(tr_w_cvt, op=cvt_f16x8_to_u4x8)

            tg_wq = tensor_view(
                wq[pid_n * bn :, pid_m * bm * 2 :],
                TensorLayout(((2, 4, 2, bm // 16), (8, 2, bn // 16)), ((1, 8, 4, 32), (m * 2, 2, m * 16))),
                "global",
            )
            txgx_wq = partition_dst(tg_wq, auto_copy())
            txrx_wq = partition_src(tr_wq, auto_copy())

            copy(auto_copy((bm, bn)), txrx_wq, txgx_wq)

    func = script_module.build()
    return func


def weight_quantization_4bits(m, n):
    from hidet.lang.types import u32, i32, f16, f32, i4, u4
    from hidet.lang import attrs
    from hidet.lang import shared_tensor, register_tensor
    from hidet.lang.cuda import syncthreads, cp_async_wait_all
    from hidet.lang.cuda import blockIdx, threadIdx, dynamic_shared_memory
    from hidet.lang.cuda import cp_async, cp_async_commit_group, cp_async_wait_group

    from hidet.ir.cute.algorithm import auto_copy, auto_mma
    from hidet.ir.cute import auto_layout

    from hidet.utils.py import cdiv
    from hidet.lang import grid
    from hidet.ir.expr import Expr, cast

    from hidet.ir.primitives import cvtv
    from hidet.ir.primitives.cuda.cvt import cvtv_func

    bm = 32
    bn = 128
    threads = 128

    def cvt_f16x8_to_u4x8(x: Expr):
        return u4(x)

    with hidet.script_module() as script_module:

        @hidet.script
        def func(w: f16[m, n], wq: u4[n, m]):
            attrs.func_kind = "cuda_kernel"
            attrs.cuda.block_dim = threads
            attrs.cuda.grid_dim = cdiv(m, bm) * cdiv(n, bn), 1
            attrs.cuda.dynamic_smem_bytes = 0

            pid = blockIdx.x
            num_pid_n = cdiv(n, bn)
            pid_m = pid // num_pid_n
            pid_n = pid % num_pid_n

            tg_w = tensor_view(w[pid_m * bm :, pid_n * bn :], TensorLayout((bm, bn), (n, 1)), "global")
            tr_w = make_tensor("float16", auto_layout, "register")

            txgx_w = partition_src(tg_w, auto_copy())
            txrx_w = partition_dst(tr_w, auto_copy())
            copy(auto_copy((bm, bn)), txgx_w, txrx_w)

            tr_w_cvt = rearrange(tr_w, auto_layout, "register")

            tr_wq = arithmetic(tr_w_cvt, op=cvt_f16x8_to_u4x8)

            tg_wq = tensor_view(
                wq[pid_n * bn :, pid_m * bm * 2 :],
                TensorLayout(((2, 4, 2, bm // 16), (8, 2, bn // 16)), ((1, 8, 2, 32), (m * 2, 4, m * 16))),
                "global",
            )
            txgx_wq = partition_dst(tg_wq, auto_copy())
            txrx_wq = partition_src(tr_wq, auto_copy())

            copy(auto_copy((bm, bn)), txrx_wq, txgx_wq)

    func = script_module.build()
    return func


def weight_dequantization_4bits(m, n):
    from hidet.lang.types import u32, i32, f16, f32, i4, u4
    from hidet.lang import attrs
    from hidet.lang import shared_tensor, register_tensor
    from hidet.lang.cuda import syncthreads, cp_async_wait_all
    from hidet.lang.cuda import blockIdx, threadIdx, dynamic_shared_memory
    from hidet.lang.cuda import cp_async, cp_async_commit_group, cp_async_wait_group

    from hidet.ir.cute.algorithm import auto_copy, auto_mma
    from hidet.ir.cute import auto_layout

    from hidet.utils.py import cdiv
    from hidet.lang import grid
    from hidet.ir.expr import Expr, cast

    from hidet.ir.primitives import cvtv
    from hidet.ir.primitives.cuda.cvt import cvtv_func

    bm = 32
    bn = 128
    threads = 128

    #    def cvt(x: Expr):
    #        return f16(x)
    cvt = cvtv_func(u4, f16)

    from hidet.ir.expr import call
    from hidet.lang import printf

    with hidet.script_module() as script_module:

        @hidet.script
        def func(wq: u4[n, m], wdq: f16[m, n]):
            attrs.func_kind = "cuda_kernel"
            attrs.cuda.block_dim = threads
            attrs.cuda.grid_dim = cdiv(m, bm) * cdiv(n, bn), 1
            attrs.cuda.dynamic_smem_bytes = 0

            pid = blockIdx.x
            num_pid_n = cdiv(n, bn)
            pid_m = pid // num_pid_n
            pid_n = pid % num_pid_n

            tg_wq = tensor_view(
                wq[pid_n * bn :, pid_m * bm * 2 :],
                TensorLayout(((2, 4, 2, bm // 16), (8, 2, bn // 16)), ((1, 8, 2, 32), (m * 2, 4, m * 16))),
                "global",
            )
            tr_wq = make_tensor("uint4b", auto_layout, "register")

            txgx_wq = partition_src(tg_wq, auto_copy())
            txrx_wq = partition_dst(tr_wq, auto_copy())
            copy(auto_copy((bm, bn)), txgx_wq, txrx_wq)

            tr_w = arithmetic(tr_wq, op=cvt)
            tr_w_1 = rearrange(tr_w, auto_layout, "register")

            tg_w = tensor_view(
                wdq[pid_m * bm : (pid_m + 1) * bm, pid_n * bn : (pid_n + 1) * bn],
                TensorLayout((bm, bn), (n, 1)),
                "global",
            )
            txgx_w = partition_dst(tg_w, auto_copy())
            txrx_w = partition_src(tr_w_1, auto_copy())

            copy(auto_copy((bm, bn)), txrx_w, txgx_w)

    func = script_module.build()
    return func


def weight_quantization_2bits(m, n):
    from hidet.lang.types import u32, i32, f16, f32, i2, u2
    from hidet.lang import attrs
    from hidet.lang import shared_tensor, register_tensor
    from hidet.lang.cuda import syncthreads, cp_async_wait_all
    from hidet.lang.cuda import blockIdx, threadIdx, dynamic_shared_memory
    from hidet.lang.cuda import cp_async, cp_async_commit_group, cp_async_wait_group

    from hidet.ir.cute.algorithm import auto_copy, auto_mma
    from hidet.ir.cute import auto_layout

    from hidet.utils.py import cdiv
    from hidet.lang import grid
    from hidet.ir.expr import Expr, cast

    from hidet.ir.primitives import cvtv
    from hidet.ir.primitives.cuda.cvt import cvtv_func

    bm = 64
    bn = 128
    threads = 128

    def cvt(x: Expr):
        return u2(x)

    with hidet.script_module() as script_module:

        @hidet.script
        def func(w: f16[m, n], wq: u2[n, m]):
            attrs.func_kind = "cuda_kernel"
            attrs.cuda.block_dim = threads
            attrs.cuda.grid_dim = cdiv(m, bm) * cdiv(n, bn), 1
            attrs.cuda.dynamic_smem_bytes = 0

            pid = blockIdx.x
            num_pid_n = cdiv(n, bn)
            pid_m = pid // num_pid_n
            pid_n = pid % num_pid_n

            tg_w = tensor_view(w[pid_m * bm :, pid_n * bn :], TensorLayout((bm, bn), (n, 1)), "global")
            tr_w = make_tensor("float16", auto_layout, "register")

            txgx_w = partition_src(tg_w, auto_copy())
            txrx_w = partition_dst(tr_w, auto_copy())
            copy(auto_copy((bm, bn)), txgx_w, txrx_w)

            tr_w_cvt = rearrange(tr_w, auto_layout, "register")

            tr_wq = arithmetic(tr_w_cvt, op=cvt)

            tg_wq = tensor_view(
                wq[pid_n * bn :, pid_m * bm * 4 :],
                TensorLayout(((2, 4, 2, bm // 16), (8, 4, bn // 32)), ((1, 16, 2, 64), (m * 4, 4, m * 32))),
                "global",
            )
            txgx_wq = partition_dst(tg_wq, auto_copy())
            txrx_wq = partition_src(tr_wq, auto_copy())

            copy(auto_copy((bm, bn)), txrx_wq, txgx_wq)

    func = script_module.build()
    return func


def weight_dequantization_2bits(m, n):
    from hidet.lang.types import u32, i32, f16, f32, i2, u2
    from hidet.lang import attrs
    from hidet.lang import shared_tensor, register_tensor
    from hidet.lang.cuda import syncthreads, cp_async_wait_all
    from hidet.lang.cuda import blockIdx, threadIdx, dynamic_shared_memory
    from hidet.lang.cuda import cp_async, cp_async_commit_group, cp_async_wait_group

    from hidet.ir.cute.algorithm import auto_copy, auto_mma
    from hidet.ir.cute import auto_layout

    from hidet.utils.py import cdiv
    from hidet.lang import grid
    from hidet.ir.expr import Expr, cast

    from hidet.ir.primitives import cvtv
    from hidet.ir.primitives.cuda.cvt import cvtv_func

    bm = 64
    bn = 128
    threads = 128

    cvt = cvtv_func(u2, f16)
    #    def cvt(x):
    #        return f16(x)

    from hidet.ir.expr import call
    from hidet.lang import printf

    with hidet.script_module() as script_module:

        @hidet.script
        def func(wq: u2[n, m], wdq: f16[m, n]):
            attrs.func_kind = "cuda_kernel"
            attrs.cuda.block_dim = threads
            attrs.cuda.grid_dim = cdiv(m, bm) * cdiv(n, bn), 1
            attrs.cuda.dynamic_smem_bytes = 0

            pid = blockIdx.x
            num_pid_n = cdiv(n, bn)
            pid_m = pid // num_pid_n
            pid_n = pid % num_pid_n

            tg_wq = tensor_view(
                wq[pid_n * bn :, pid_m * bm * 4 :],
                TensorLayout(((2, 4, 2, bm // 16), (8, 4, bn // 32)), ((1, 16, 2, 64), (m * 4, 4, m * 32))),
                "global",
            )
            tr_wq = make_tensor("uint2b", auto_layout, "register")

            txgx_wq = partition_src(tg_wq, auto_copy())
            txrx_wq = partition_dst(tr_wq, auto_copy())
            copy(auto_copy((bm, bn)), txgx_wq, txrx_wq)

            tr_w = arithmetic(tr_wq, op=cvt)
            tr_w_1 = rearrange(tr_w, auto_layout, "register")

            tg_w = tensor_view(
                wdq[pid_m * bm : (pid_m + 1) * bm, pid_n * bn : (pid_n + 1) * bn],
                TensorLayout((bm, bn), (n, 1)),
                "global",
            )
            txgx_w = partition_dst(tg_w, auto_copy())
            txrx_w = partition_src(tr_w_1, auto_copy())

            copy(auto_copy((bm, bn)), txrx_w, txgx_w)

    func = script_module.build()
    return func


def weight_quantization_1bits(m, n):
    from hidet.lang.types import u32, i32, f16, f32, u1
    from hidet.lang import attrs
    from hidet.lang import shared_tensor, register_tensor
    from hidet.lang.cuda import syncthreads, cp_async_wait_all
    from hidet.lang.cuda import blockIdx, threadIdx, dynamic_shared_memory
    from hidet.lang.cuda import cp_async, cp_async_commit_group, cp_async_wait_group

    from hidet.ir.cute.algorithm import auto_copy, auto_mma
    from hidet.ir.cute import auto_layout

    from hidet.utils.py import cdiv
    from hidet.lang import grid
    from hidet.ir.expr import Expr, cast

    from hidet.ir.primitives import cvtv
    from hidet.ir.primitives.cuda.cvt import cvtv_func

    bm = 32
    bn = 128
    threads = 128

    def cvt(x: Expr):
        return u1(x)

    with hidet.script_module() as script_module:

        @hidet.script
        def func(w: f16[m, n], wq: u1[n, m]):
            attrs.func_kind = "cuda_kernel"
            attrs.cuda.block_dim = threads
            attrs.cuda.grid_dim = cdiv(m, bm) * cdiv(n, bn), 1
            attrs.cuda.dynamic_smem_bytes = 0

            pid = blockIdx.x
            num_pid_n = cdiv(n, bn)
            pid_m = pid // num_pid_n
            pid_n = pid % num_pid_n

            tg_w = tensor_view(w[pid_m * bm :, pid_n * bn :], TensorLayout((bm, bn), (n, 1)), "global")
            tr_w = make_tensor("float16", auto_layout, "register")

            txgx_w = partition_src(tg_w, auto_copy())
            txrx_w = partition_dst(tr_w, auto_copy())
            copy(auto_copy((bm, bn)), txgx_w, txrx_w)

            tr_w_cvt = rearrange(tr_w, auto_layout, "register")

            tr_wq = arithmetic(tr_w_cvt, op=cvt)

            tg_wq = tensor_view(
                wq[pid_n * bn :, pid_m * bm * 4 :],
                TensorLayout(((2, 4, 2, bm // 16), (8, 4, bn // 32)), ((1, 16, 2, 64), (m * 4, 4, m * 32))),
                "global",
            )
            txgx_wq = partition_dst(tg_wq, auto_copy())
            txrx_wq = partition_src(tr_wq, auto_copy())

            copy(auto_copy((bm, bn)), txrx_wq, txgx_wq)

    func = script_module.build()
    return func


def weight_dequantization_1bits(m, n):
    from hidet.lang.types import u32, i32, f16, f32, u1
    from hidet.lang import attrs
    from hidet.lang import shared_tensor, register_tensor
    from hidet.lang.cuda import syncthreads, cp_async_wait_all
    from hidet.lang.cuda import blockIdx, threadIdx, dynamic_shared_memory
    from hidet.lang.cuda import cp_async, cp_async_commit_group, cp_async_wait_group

    from hidet.ir.cute.algorithm import auto_copy, auto_mma
    from hidet.ir.cute import auto_layout

    from hidet.utils.py import cdiv
    from hidet.lang import grid
    from hidet.ir.expr import Expr, cast

    from hidet.ir.primitives import cvtv
    from hidet.ir.primitives.cuda.cvt import cvtv_func

    bm = 64
    bn = 128
    threads = 128

    cvt = cvtv_func(u2, f16)
    #    def cvt(x):
    #        return f16(x)

    from hidet.ir.expr import call
    from hidet.lang import printf

    with hidet.script_module() as script_module:

        @hidet.script
        def func(wq: u1[n, m], wdq: f16[m, n]):
            attrs.func_kind = "cuda_kernel"
            attrs.cuda.block_dim = threads
            attrs.cuda.grid_dim = cdiv(m, bm) * cdiv(n, bn), 1
            attrs.cuda.dynamic_smem_bytes = 0

            pid = blockIdx.x
            num_pid_n = cdiv(n, bn)
            pid_m = pid // num_pid_n
            pid_n = pid % num_pid_n

            tg_wq = tensor_view(
                wq[pid_n * bn :, pid_m * bm * 4 :],
                TensorLayout(((2, 4, 2, bm // 16), (8, 4, bn // 32)), ((1, 16, 2, 64), (m * 4, 4, m * 32))),
                "global",
            )
            tr_wq = make_tensor("uint2b", auto_layout, "register")

            txgx_wq = partition_src(tg_wq, auto_copy())
            txrx_wq = partition_dst(tr_wq, auto_copy())
            copy(auto_copy((bm, bn)), txgx_wq, txrx_wq)

            tr_w = arithmetic(tr_wq, op=cvt)
            tr_w_1 = rearrange(tr_w, auto_layout, "register")

            tg_w = tensor_view(
                wdq[pid_m * bm : (pid_m + 1) * bm, pid_n * bn : (pid_n + 1) * bn],
                TensorLayout((bm, bn), (n, 1)),
                "global",
            )
            txgx_w = partition_dst(tg_w, auto_copy())
            txrx_w = partition_src(tr_w_1, auto_copy())

            copy(auto_copy((bm, bn)), txrx_w, txgx_w)

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


def data(M, N, lo=-3, hi=3, dtype="float16", device="cuda", return_hidet=False):
    dtype = getattr(torch, dtype)
    a = torch.randint(low=lo, high=hi, size=(M, N), dtype=dtype, device=device)
    if return_hidet:
        a = hidet.from_torch(a)
    return a


@pytest.mark.requires_cuda
@pytest.mark.parametrize("m,n", [(4096, 4096 * 3), (4096 * 3, 4096)])
def test_quant_4bits(m, n):
    qfunc = weight_quantization_4bits(m, n)
    w = data(m, n, lo=0, hi=16, return_hidet=True)
    wq = data(m, n // 2, lo=0, hi=16, dtype="uint8", return_hidet=True)
    wdq = data(m, n, lo=0, hi=16, return_hidet=True)
    mean, min_lat, max_lat = bench(qfunc, (w, wq))
    print("time={:.3f} ms, performance={:.3f} GB/s".format(mean, (m * n * 2 + m * n // 2) / (1e6 * mean)))

    dqfunc = weight_dequantization_4bits(m, n)
    mean, min_lat, max_lat = bench(dqfunc, (wq, wdq))
    print("time={:.3f} ms, performance={:.3f} GB/s".format(mean, (m * n * 2 + m * n // 2) / (1e6 * mean)))
    dqfunc(wq, wdq)

    import numpy as np

    np.set_printoptions(threshold=3000, linewidth=200, edgeitems=100)
    np.testing.assert_allclose(actual=w.cpu().numpy(), desired=wdq.cpu().numpy(), rtol=1e-2)


@pytest.mark.requires_cuda
@pytest.mark.parametrize("m,n", [(4096, 4096 * 3), (4096 * 3, 4096)])
def test_quant_2bits(m, n):
    qfunc = weight_quantization_2bits(m, n)
    w = data(m, n, lo=0, hi=4, return_hidet=True)
    wq = data(m, n // 4, lo=0, hi=4, dtype="uint8", return_hidet=True)
    wdq = data(m, n, lo=0, hi=4, return_hidet=True)
    mean, min_lat, max_lat = bench(qfunc, (w, wq))
    print("time={:.3f} ms, performance={:.3f} GB/s".format(mean, (m * n * 2 + m * n // 4) / (1e6 * mean)))

    dqfunc = weight_dequantization_2bits(m, n)
    mean, min_lat, max_lat = bench(dqfunc, (wq, wdq))
    print("time={:.3f} ms, performance={:.3f} GB/s".format(mean, (m * n * 2 + m * n // 4) / (1e6 * mean)))
    dqfunc(wq, wdq)

    import numpy as np

    np.set_printoptions(threshold=3000, linewidth=200, edgeitems=100)
    np.testing.assert_allclose(actual=w.cpu().numpy(), desired=wdq.cpu().numpy(), rtol=1e-2)


if __name__ == "__main__":
    hidet.option.cache_dir("./demo_weight_quant")
    hidet.option.search_space(2)
    hidet.option.debug_cache_tuning()
    hidet.option.save_lower_ir(True)

    test_quant_4bits(4096, 4096 * 3)
    test_quant_2bits(4096 * 3, 4096)
