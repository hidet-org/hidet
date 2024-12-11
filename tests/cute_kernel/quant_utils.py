from typing import Tuple
import hidet

from hidet.lang import attrs
from hidet.lang.cuda import blockIdx

from hidet.ir.cute.layout import TensorLayout, make_layout
from hidet.ir.cute.algorithm import auto_copy
from hidet.ir.cute.ops import make_tensor, tensor_view, partition_src, partition_dst, copy, rearrange, cast

from hidet.ir.cute import auto_layout
from hidet.ir.cute import composition, coalesce, logical_divide

from hidet.utils.py import cdiv

from hidet.lang.types import i32, f16, u4
from hidet.ir.type import DataType

from hidet.ir.module import IRModule
import torch


class GemmQuantModule:
    def __init__(self, gemm_quant: IRModule, q: Tuple[IRModule, ...], dq: Tuple[IRModule, ...]):
        self.gemm_quant = gemm_quant
        self.q = q
        self.dq = dq

    @property
    def module(self):
        return self.gemm_quant

    @property
    def quant(self):
        return self.q

    @property
    def dequant(self):
        return self.dq


def gemm_quant_module(gemm_quant: IRModule, q: Tuple[IRModule, ...], dq: Tuple[IRModule, ...]):
    return GemmQuantModule(gemm_quant, q, dq)


def weight_quantization_subbyte(m, n, gmem_layout: TensorLayout, wdtype: DataType = u4):
    bm = 128
    bn = 256
    threads = 256

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
        def func(w: f16[m, n], wq: wdtype[n, m]):
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

            tr_wq = cast(tr_w_cvt, wdtype)

            tg_wq = tensor_view(wq[pid_n * bn * n_stride :, pid_m * bm * m_stride :], gmem, "global")
            txgx_wq = partition_dst(tg_wq, auto_copy())
            txrx_wq = partition_src(tr_wq, auto_copy())

            copy(auto_copy((bm, bn)), txrx_wq, txgx_wq)

        return script_module.ir_module()


def weight_dequantization_subbyte(m, n, gmem_layout: TensorLayout, wdtype: DataType = u4):
    bm = 128
    bn = 256
    threads = 256

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
        def func(wq: wdtype[n, m], wdq: f16[m, n]):
            attrs.func_kind = "cuda_kernel"
            attrs.cuda.block_dim = threads
            attrs.cuda.grid_dim = cdiv(m, bm) * cdiv(n, bn), 1
            attrs.cuda.dynamic_smem_bytes = 0

            pid = blockIdx.x
            num_pid_n = cdiv(n, bn)
            pid_m = pid // num_pid_n
            pid_n = pid % num_pid_n

            tg_wq = tensor_view(wq[pid_n * bn * n_stride :, pid_m * bm * m_stride :], gmem, "global")
            tr_wq = make_tensor(wdtype, auto_layout, "register")

            txgx_wq = partition_src(tg_wq, auto_copy())
            txrx_wq = partition_dst(tr_wq, auto_copy())
            copy(auto_copy((bm, bn)), txgx_wq, txrx_wq)

            tr_w = cast(tr_wq, f16)
            tr_w_1 = rearrange(tr_w, auto_layout, "register")

            tg_w = tensor_view(
                wdq[pid_m * bm : (pid_m + 1) * bm, pid_n * bn : (pid_n + 1) * bn],
                TensorLayout((bm, bn), (n, 1)),
                "global",
            )
            txgx_w = partition_dst(tg_w, auto_copy())
            txrx_w = partition_src(tr_w_1, auto_copy())

            copy(auto_copy((bm, bn)), txrx_w, txgx_w)

        return script_module.ir_module()


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


def data(M, N, K, dtype="float16", device="cuda", return_hidet=False, group_size=128):
    dtype = getattr(torch, dtype)
    lo = -2
    hi = 2
    a = torch.randint(low=lo, high=hi, size=(M, K), dtype=dtype, device=device)
    b2 = torch.randint(low=0, high=2, size=(K, N), dtype=dtype, device=device)
    b1 = torch.randint(low=0, high=2, size=(K, N), dtype=dtype, device=device)
    c = torch.empty((M, N), dtype=dtype, device=device)
    bq2 = torch.randint(low=0, high=2, size=(K, N // 4), dtype=torch.uint8, device=device)
    bq1 = torch.randint(low=0, high=2, size=(K, N // 8), dtype=torch.uint8, device=device)
    bdq = torch.randint(low=0, high=1, size=(K, N), dtype=dtype, device=device)
    scale = torch.randint(low=lo, high=hi, size=(K // group_size, N), dtype=dtype, device=device)
    zeros = torch.randint(low=lo, high=0, size=(K // group_size, N), dtype=dtype, device=device)

    a = a / K
    ret = [a, b2, b1, c, bq2, bq1, bdq, scale, zeros]

    if return_hidet:
        ret = [hidet.from_torch(x) for x in ret]

    return tuple(ret)


def canonicalize(layout: TensorLayout):
    return layout[0][0], coalesce(make_layout(layout[0][1], layout[1]))
