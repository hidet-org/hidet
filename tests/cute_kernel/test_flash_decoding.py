import hidet
import torch
import pytest
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
)
from hidet.lang.mapping import auto_map
from hidet.ir.primitives.cuda.mutex import release_seq_semaphore, acquire_seq_semaphore
from hidet.ir.primitives.cuda.atomic import atomic_add, atomic_sub
from hidet.utils.py import cdiv

from quant_utils import canonicalize, bench


def get_tiled_mma(head_size):
    if head_size == 128:
        a = TensorLayout(((4, 8), (2, 2, 2)), ((32, 1), (16, 8, 128)))
        b = TensorLayout(((4, 8), (2, 2)), ((16, 1), (8, 64)))
        c = TensorLayout(((4, 8), (2, 2)), ((32, 1), (16, 8)))
        mma_atom = MmaAtom("warp", (16, 8, 16), a, b, c, c, (1, 2))
        warp_in_threadblock = Level("warp", "thread_block", (4, 1), TensorLayout((4, 1)), (1, 8))
        tiled_mma = TiledMma(mma_atom, [warp_in_threadblock])
        return tiled_mma
    elif head_size == 64:
        a = TensorLayout(((4, 8), (2, 2, 2)), ((32, 1), (16, 8, 128)))
        b = TensorLayout(((4, 8), (2, 2)), ((16, 1), (8, 64)))
        c = TensorLayout(((4, 8), (2, 2)), ((32, 1), (16, 8)))
        mma_atom = MmaAtom("warp", (16, 8, 16), a, b, c, c, (1, 2))
        warp_in_threadblock = Level("warp", "thread_block", (4, 1), TensorLayout((4, 1)), (2, 4))
        tiled_mma = TiledMma(mma_atom, [warp_in_threadblock])
        return tiled_mma
    elif head_size == 32:
        a = TensorLayout(((4, 8), (2, 2, 2)), ((32, 1), (16, 8, 128)))
        b = TensorLayout(((4, 8), (2, 2)), ((16, 1), (8, 64)))
        c = TensorLayout(((4, 8), (2, 2)), ((32, 1), (16, 8)))
        mma_atom = MmaAtom("warp", (16, 8, 16), a, b, c, c, (1, 2))
        warp_in_threadblock = Level("warp", "thread_block", (8, 1), TensorLayout((8, 1)), (1, 2))
        tiled_mma = TiledMma(mma_atom, [warp_in_threadblock])
        return tiled_mma
    else:
        raise NotImplementedError()


def flash_decoding_v1(batch_size, seqlen_q, num_heads, head_size, seqlen_k, num_heads_k):
    from hidet.lang.types import u32, i32, f16, f32
    from hidet.lang import attrs
    from hidet.lang import shared_tensor, register_tensor
    from hidet.lang.cuda import syncthreads, cp_async_wait_all
    from hidet.lang.cuda import blockIdx, threadIdx, dynamic_shared_memory
    from hidet.lang.cuda import cp_async, cp_async_commit_group, cp_async_wait_group

    # bm * bn
    a = TensorLayout(((4, 8), (2, 2)), ((16, 1), (8, 64)))
    b = TensorLayout(((4, 8), (2, 2, 2)), ((32, 1), (16, 8, 128)))
    c = TensorLayout(((4, 8), (2, 2)), ((2, 8), (1, 64)))
    mma_atom = MmaAtom("warp", (8, 16, 16), a, b, c, c, (1, 1))
    warp_in_threadblock = Level("warp", "thread_block", (1, 4), TensorLayout((1, 4)), (1, 1))
    tiled_mma_qk = TiledMma(mma_atom, [warp_in_threadblock])

    # a = TensorLayout(((4, 8), (2, 2, 2)), ((32, 1), (16, 8, 128)))
    # b = TensorLayout(((4, 8), (2, 2)), ((16, 1), (8, 64)))
    # c = TensorLayout(((4, 8), (2, 2)), ((32, 1), (16, 8)))
    # mma_atom = MmaAtom("warp", (16, 8, 16), a, b, c, c, (1, 2))
    # warp_in_threadblock = Level(
    #    "warp", "thread_block", (2, 2), TensorLayout((2, 2)), (4, 2)
    # )
    # tiled_mma_qk = TiledMma(mma_atom, [warp_in_threadblock])

    print(tiled_mma_qk.str_indented())
    q_shape, q_tv_layout = tiled_mma_qk.a_tv_layout()
    k_shape, k_tv_layout = tiled_mma_qk.b_tv_layout()
    qk_shape, qk_tv_layout = tiled_mma_qk.c_tv_layout()
    print(q_shape, q_tv_layout)
    print(k_shape, k_tv_layout)
    print(qk_shape, qk_tv_layout)

    _, q_v = canonicalize(q_tv_layout)
    _, k_v = canonicalize(k_tv_layout)
    qk_t, qk_v = canonicalize(qk_tv_layout)

    threads = qk_t.size()

    q_elements = q_v.size()
    k_elements = k_v.size()
    qk_elements = qk_v.size()

    bm, inst_h = q_shape
    bn, inst_h_ = k_shape
    bm_, bn_ = qk_shape
    assert bm == bm_ and bn == bn_ and inst_h == inst_h_
    bk = 32

    a = TensorLayout(((4, 8), (2, 2)), ((16, 1), (8, 64)))
    b = TensorLayout(((4, 8), (2, 2, 2)), ((32, 1), (16, 8, 128)))
    c = TensorLayout(((4, 8), (2, 2)), ((2, 8), (1, 64)))
    mma_atom = MmaAtom("warp", (8, 16, 16), a, b, c, c, (1, 1))
    warp_in_threadblock = Level("warp", "thread_block", (1, 4), TensorLayout((1, 4)), (1, 2))
    tiled_mma_o = TiledMma(mma_atom, [warp_in_threadblock])

    # a = TensorLayout(((4, 8), (2, 2, 2)), ((32, 1), (16, 8, 128)))
    # b = TensorLayout(((4, 8), (2, 2)), ((16, 1), (8, 64)))
    # c = TensorLayout(((4, 8), (2, 2)), ((32, 1), (16, 8)))
    # mma_atom = MmaAtom("warp", (16, 8, 16), a, b, c, c, (1, 2))
    # warp_in_threadblock = Level(
    #    "warp", "thread_block", (2, 2), TensorLayout((2, 2)), (4, 4)
    # )
    # tiled_mma_o = TiledMma(mma_atom, [warp_in_threadblock])

    print(tiled_mma_o.str_indented())
    qk_shape, qk_tv_layout = tiled_mma_o.a_tv_layout()
    v_shape, v_tv_layout = tiled_mma_o.b_tv_layout()
    o_shape, o_tv_layout = tiled_mma_o.c_tv_layout()
    print(qk_shape, qk_tv_layout)
    print(v_shape, v_tv_layout)
    print(o_shape, o_tv_layout)

    _, inst_n = qk_shape

    _, qk_v = canonicalize(qk_tv_layout)
    _, v_v = canonicalize(v_tv_layout)
    _, o_v = canonicalize(o_tv_layout)

    qk1_elements = qk_v.size()
    v_elements = v_v.size()
    o_elements = o_v.size()

    from hidet.ir.cute.algorithm import auto_copy, auto_mma
    from hidet.ir.cute import auto_layout, layout_auto

    stages = 2
    smem_gemm_qk = (bm * bk + bn * bk) * stages * f16.nbytes
    smem_gmem_o = (bm * bn + bn * head_size) * f16.nbytes
    dynamic_smem_bytes = max(smem_gemm_qk, smem_gmem_o)

    from hidet.lang import grid

    n_size = seqlen_q * num_heads
    bs = batch_size * num_heads
    assert num_heads == num_heads_k
    num_pid_m = cdiv(seqlen_q, bm)
    num_pid_n = cdiv(seqlen_k, bn)
    #    assert head_size == bn
    import sys

    float_max = f32.max_value

    qk_max_elements = 4
    qk_sum_elements = 4

    with hidet.script_module() as script_module:

        @hidet.script
        def func(
            q: f16[batch_size, seqlen_q, num_heads, head_size],
            k: f16[batch_size, seqlen_k, num_heads_k, head_size],
            v: f16[batch_size, seqlen_k, num_heads_k, head_size],
            o: f16[batch_size, seqlen_q, num_heads, head_size],
            lock: i32[batch_size, num_heads, num_pid_m],
            imm: f32[batch_size, num_heads, num_pid_n, seqlen_q],
        ):
            attrs.func_kind = "cuda_kernel"
            attrs.cuda.block_dim = threads
            attrs.cuda.grid_dim = num_pid_m * num_pid_n, bs
            attrs.cuda.dynamic_smem_bytes = dynamic_smem_bytes

            pid = blockIdx.x
            pid_m = pid // num_pid_n
            pid_n = pid % num_pid_n

            bs_idx = blockIdx.y
            batch_idx = bs_idx // num_heads
            head_idx = bs_idx % num_heads

            if pid_n == 0 and threadIdx.x == 0:
                lock[batch_idx, head_idx, pid_m] = 0

            tr_q = make_tensor("float16", layout_auto((bm, inst_h * 2)), "register")
            tr_k = make_tensor("float16", layout_auto((bn, inst_h * 2)), "register")
            tr_qk = make_tensor("float32", auto_layout, "register")

            fill(tr_qk, 0.0)

            tg_q = tensor_view(
                q[batch_idx, pid_m * bm :, head_idx, :],
                TensorLayout((bm, head_size), (num_heads * head_size, 1)),
                "global",
            )
            tg_k = tensor_view(
                k[batch_idx, pid_n * bn :, head_idx, :],
                TensorLayout((bn, head_size), (num_heads_k * head_size, 1)),
                "global",
            )

            txgq = partition_src(tg_q, auto_copy())
            txgk = partition_src(tg_k, auto_copy())

            ts_q = make_tensor("float16", TensorLayout((bm, bk, stages), (bk, 1, bm * bk)), "shared")
            ts_k = make_tensor("float16", TensorLayout((bn, bk, stages), (bk, 1, bn * bk)), "shared")

            txsq = partition_dst(ts_q, auto_copy())
            txsk = partition_dst(ts_k, auto_copy())

            for s in range(stages - 1):
                copy(auto_copy((bm, bk)), txgq[:, :, s], txsq[:, :, s])
                copy(auto_copy((bn, bk)), txgk[:, :, s], txsk[:, :, s])
                cp_async_commit_group()

            smem_pipe_read = 0
            smem_pipe_write = stages - 1

            txSq = partition_src(ts_q, auto_copy())
            txrq = partition_dst(tr_q, auto_copy())

            txSk = partition_src(ts_k, auto_copy())
            txrk = partition_dst(tr_k, auto_copy())

            txSq_p = txSq[:, :, :, smem_pipe_read]
            txSk_p = txSk[:, :, :, smem_pipe_read]

            cp_async_wait_group(allow_on_fly_groups=stages - 2)
            syncthreads()

            copy(auto_copy(), txSq_p[:, :, 0], txrq[:, :, 0])
            copy(auto_copy(), txSk_p[:, :, 0], txrk[:, :, 0])

            h_tile_max = (bk + inst_h - 1) // inst_h
            for ho in range((head_size + bk - 1) // bk):
                for hi in range(h_tile_max):
                    if hi == h_tile_max - 1:
                        cp_async_wait_group(allow_on_fly_groups=0)
                        syncthreads()

                    h_tile_next = (hi + 1) % h_tile_max
                    copy(auto_copy(), txSq[:, :, h_tile_next, smem_pipe_read], txrq[:, :, (hi + 1) % 2])
                    copy(auto_copy(), txSk[:, :, h_tile_next, smem_pipe_read], txrk[:, :, (hi + 1) % 2])
                    if hi == 0:
                        if ho + stages - 1 < (head_size + bk - 1) // bk:
                            copy(auto_copy((bm, bk)), txgq[:, :, ho + stages - 1], txsq[:, :, smem_pipe_write])
                            copy(auto_copy((bn, bk)), txgk[:, :, ho + stages - 1], txsk[:, :, smem_pipe_write])
                        smem_pipe_write = smem_pipe_read
                        cp_async_commit_group()

                    if hi == h_tile_max - 2:
                        smem_pipe_read += 1
                        smem_pipe_read = 0 if smem_pipe_read == stages else smem_pipe_read

                    mma(tiled_mma_qk, tr_qk, txrq[:, :, hi % 2], txrk[:, :, hi % 2], tr_qk)

            lc = ~lock[batch_idx, head_idx, pid_m]
            tr_qk_max = reduce_max(tr_qk, axis=1)

            if pid_n < num_pid_n - 1:
                tg_qk_max = tensor_view(
                    imm[batch_idx, head_idx, pid_n, pid_m * bm :], TensorLayout((bm, bn), (1, 0)), "global"
                )
                txrqk_max = partition_src(tr_qk_max, auto_copy())
                txgqk_max = partition_dst(tg_qk_max, auto_copy())
                copy(auto_copy((bm, bn)), txrqk_max, txgqk_max)
            #    syncthreads()
            #    if threadIdx.x == 0:
            #        atomic_add(lc, 1)
            # else:
            #    qk_max_part = register_tensor("float32", shape=[qk_max_elements])
            #    tr_qk_max_part = tensor_view(qk_max_part, auto_layout, "register")
            #    acquire_seq_semaphore(lc, pid_n)
            #    for i in range(num_pid_n - 1):
            #        tg_qk_max = tensor_view(imm[batch_idx, head_idx, i, pid_m * bm:], TensorLayout((bm, bn), (1, 0)), "global")
            #        txgqk_max = partition_src(tg_qk_max, auto_copy())
            #        txrqk_max = partition_dst(tr_qk_max_part, auto_copy())
            #        copy(auto_copy((bm, bn)), txgqk_max, txrqk_max)
            #        tr_qk_max = elementwise_max(tr_qk_max, tr_qk_max_part)
            #    tg_qk_max = tensor_view(imm[batch_idx, head_idx, pid_n, pid_m * bm:], TensorLayout((bm, bn), (1, 0)), "global")
            #    txrqk_max_final = partition_src(tr_qk_max, auto_copy())
            #    txgqk_max = partition_dst(tg_qk_max, auto_copy())
            #    copy(auto_copy((bm, bn)), txrqk_max_final, txgqk_max)
            #    release_seq_semaphore(lc, pid_n + 1)
            # acquire_seq_semaphore(lc, num_pid_n)
            # tg_qk_max = tensor_view(imm[batch_idx, head_idx, pid_n, pid_m * bm:], TensorLayout((bm, bn), (1, 0)), "global")
            # txgqk_max = partition_src(tg_qk_max, auto_copy())
            # txrqk_max = partition_dst(tr_qk_max, auto_copy())
            # copy(auto_copy((bm, bn)), txgqk_max, txrqk_max)

            tr_qk_exp = exp(tr_qk - tr_qk_max)
            tr_qk_sum = reduce_sum(tr_qk_exp, axis=1)
            if pid_n < num_pid_n - 1:
                tg_qk_sum = tensor_view(
                    imm[batch_idx, head_idx, pid_n, pid_m * bm :], TensorLayout((bm, bn), (1, 0)), "global"
                )
                txrqk_sum = partition_src(tr_qk_sum, auto_copy())
                txgqk_sum = partition_dst(tg_qk_sum, auto_copy())
                copy(auto_copy((bm, bn)), txrqk_sum, txgqk_sum)
            #    syncthreads()
            #    if threadIdx.x == 0:
            #        atomic_sub(lc, 1)
            # else:
            #    qk_sum_part = register_tensor("float32", shape=[qk_sum_elements])
            #    tr_qk_sum_part = tensor_view(qk_sum_part, auto_layout, "register")
            #    acquire_seq_semaphore(lc, 1)
            #    for i in range(num_pid_n - 1):
            #        tg_qk_sum = tensor_view(imm[batch_idx, head_idx, i, pid_m * bm:], TensorLayout((bm, bn), (1, 0)), "global")
            #        txgqk_sum = partition_src(tg_qk_sum, auto_copy())
            #        txrqk_sum = partition_dst(tr_qk_sum_part, auto_copy())
            #        copy(auto_copy((bm, bn)), txgqk_sum, txrqk_sum)
            #        tr_qk_sum = tr_qk_sum + tr_qk_sum_part
            #    tg_qk_sum = tensor_view(imm[batch_idx, head_idx, pid_n, pid_m * bm:], TensorLayout((bm, bn), (1, 0)), "global")
            #    txrqk_sum_final = partition_src(tr_qk_sum, auto_copy())
            #    txgqk_sum = partition_dst(tg_qk_sum, auto_copy())
            #    copy(auto_copy((bm, bn)), txrqk_sum_final, txgqk_sum)
            #    release_seq_semaphore(lc, 0)
            # acquire_seq_semaphore(lc, 0)
            # tg_qk_sum = tensor_view(imm[batch_idx, head_idx, pid_n, pid_m * bm:], TensorLayout((bm, bn), (1, 0)), "global")
            # txgqk_sum = partition_src(tg_qk_sum, auto_copy())
            # txrqk_sum = partition_dst(tr_qk_sum, auto_copy())
            # copy(auto_copy((bm, bn)), txgqk_sum, txrqk_sum)

            tr_qk_o = tr_qk_exp / tr_qk_sum
            tr_qk_f16 = cast(tr_qk_o, f16)

            ts_qk = make_tensor("float16", TensorLayout((bm, bn), (bn, 1)), "shared")

            txrqk1 = partition_src(tr_qk_f16, auto_copy())
            txsqk = partition_dst(ts_qk, auto_copy())
            copy(auto_copy((bm, bn)), txrqk1, txsqk)
            syncthreads()

            ts_v = make_tensor("float16", TensorLayout((head_size, bn), (1, head_size)), "shared")
            tr_qk1 = make_tensor("float16", layout_auto((bm, inst_n * 2)), "register")
            tr_v = make_tensor("float16", layout_auto((head_size, inst_n * 2)), "register")
            tr_o = make_tensor("float32", auto_layout, "register")

            fill(tr_o, 0.0)

            txSqk = partition_src(ts_qk, auto_copy())
            txrqk = partition_dst(tr_qk1, auto_copy())
            txSv = partition_src(ts_v, auto_copy())
            txrv = partition_dst(tr_v, auto_copy())

            copy(auto_copy(), txSqk[:, :, 0], txrqk[:, :, 0])
            copy(auto_copy(), txSv[:, :, 0], txrv[:, :, 0])

            n_tile_max = (bn + inst_n - 1) // inst_n
            for ni in range(n_tile_max):
                if ni < n_tile_max - 1:
                    copy(auto_copy(), txSqk[:, :, ni + 1], txrqk[:, :, (ni + 1) % 2])
                    copy(auto_copy(), txSv[:, :, ni + 1], txrv[:, :, (ni + 1) % 2])
                mma(tiled_mma_o, tr_o, txrqk[:, :, ni % 2], txrv[:, :, ni % 2], tr_o)

            tr_o_f16 = cast(tr_o, f16)
            tr_O = rearrange(tr_o_f16, auto_layout, "register")

            if pid_n == 0:
                tg_o = tensor_view(
                    o[batch_idx, pid_m * bm :, head_idx, 0:],
                    TensorLayout((bm, head_size), (num_heads * head_size, 1)),
                    "global",
                )
                txro = partition_src(tr_O, auto_copy())
                txgo = partition_dst(tg_o, auto_copy())
                copy(auto_copy((bm, head_size)), txro, txgo)
                # release_seq_semaphore(lc, 1)
            else:
                o_part = register_tensor("float16", shape=[o_elements])
                tr_o_part = tensor_view(o_part, auto_layout, "register")
                # acquire_seq_semaphore(lc, pid_n)
                tg_o = tensor_view(
                    o[batch_idx, pid_m * bm :, head_idx, 0:],
                    TensorLayout((bm, head_size), (num_heads * head_size, 1)),
                    "global",
                )
                txgo = partition_src(tg_o, auto_copy())
                txro_part = partition_dst(tr_o_part, auto_copy())
                copy(auto_copy((bm, head_size)), txgo, txro_part)
                tr_O = tr_O + tr_o_part
                txro_final = partition_src(tr_O, auto_copy())
                txgo1 = partition_dst(tg_o, auto_copy())
                copy(auto_copy((bm, head_size)), txro_final, txgo1)
                # release_seq_semaphore(lc, pid_n + 1)

    func = script_module.build()
    # a_mem = hidet.empty([m, k], device="cuda")
    # b_mem = hidet.empty([k, n], device="cuda")
    # c_mem = hidet.empty([m, n], device="cuda")
    # func(a_mem, b_mem, c_mem)
    return func


def flash_decoding_v2(batch_size, seqlen_q, num_heads, head_size, seqlen_k, num_heads_k):
    from hidet.lang.types import u32, i32, f16, f32
    from hidet.lang import attrs
    from hidet.lang import shared_tensor, register_tensor
    from hidet.lang.cuda import syncthreads, cp_async_wait_all
    from hidet.lang.cuda import blockIdx, threadIdx, dynamic_shared_memory
    from hidet.lang.cuda import cp_async, cp_async_commit_group, cp_async_wait_group

    # bm * bn
    a = TensorLayout(((4, 8), (2, 2)), ((16, 1), (8, 64)))
    b = TensorLayout(((4, 8), (2, 2, 2)), ((32, 1), (16, 8, 128)))
    c = TensorLayout(((4, 8), (2, 2)), ((2, 8), (1, 64)))
    mma_atom = MmaAtom("warp", (8, 16, 16), a, b, c, c, (1, 1))
    warp_in_threadblock = Level("warp", "thread_block", (1, 4), TensorLayout((1, 4)), (1, 1))
    tiled_mma_qk = TiledMma(mma_atom, [warp_in_threadblock])

    print(tiled_mma_qk.str_indented())
    q_shape, q_tv_layout = tiled_mma_qk.a_tv_layout()
    k_shape, k_tv_layout = tiled_mma_qk.b_tv_layout()
    qk_shape, qk_tv_layout = tiled_mma_qk.c_tv_layout()
    print(q_shape, q_tv_layout)
    print(k_shape, k_tv_layout)
    print(qk_shape, qk_tv_layout)

    _, q_v = canonicalize(q_tv_layout)
    _, k_v = canonicalize(k_tv_layout)
    qk_t, qk_v = canonicalize(qk_tv_layout)

    threads = qk_t.size()

    q_elements = q_v.size()
    k_elements = k_v.size()
    qk_elements = qk_v.size()

    bm, inst_h = q_shape
    bn, inst_h_ = k_shape
    bm_, bn_ = qk_shape
    assert bm == bm_ and bn == bn_ and inst_h == inst_h_
    bk = 32

    a = TensorLayout(((4, 8), (2, 2)), ((16, 1), (8, 64)))
    b = TensorLayout(((4, 8), (2, 2, 2)), ((32, 1), (16, 8, 128)))
    c = TensorLayout(((4, 8), (2, 2)), ((2, 8), (1, 64)))
    mma_atom = MmaAtom("warp", (8, 16, 16), a, b, c, c, (1, 1))
    warp_in_threadblock = Level("warp", "thread_block", (1, 4), TensorLayout((1, 4)), (1, 2))
    tiled_mma_o = TiledMma(mma_atom, [warp_in_threadblock])

    print(tiled_mma_o.str_indented())
    qk_shape, qk_tv_layout = tiled_mma_o.a_tv_layout()
    v_shape, v_tv_layout = tiled_mma_o.b_tv_layout()
    o_shape, o_tv_layout = tiled_mma_o.c_tv_layout()
    print(qk_shape, qk_tv_layout)
    print(v_shape, v_tv_layout)
    print(o_shape, o_tv_layout)

    _, inst_n = qk_shape

    _, qk_v = canonicalize(qk_tv_layout)
    _, v_v = canonicalize(v_tv_layout)
    _, o_v = canonicalize(o_tv_layout)

    qk1_elements = qk_v.size()
    v_elements = v_v.size()
    o_elements = o_v.size()

    from hidet.ir.cute.algorithm import auto_copy, auto_mma
    from hidet.ir.cute import auto_layout, layout_auto

    stages = 1
    smem_gemm_qk = (bm * bk + bn * bk) * stages * f16.nbytes
    smem_gmem_o = (bm * bn + bn * head_size) * f16.nbytes
    dynamic_smem_bytes = max(smem_gemm_qk, smem_gmem_o)

    from hidet.lang import grid

    n_size = seqlen_q * num_heads
    bs = batch_size * num_heads
    assert num_heads == num_heads_k
    num_pid_m = cdiv(seqlen_q, bm)
    num_pid_n = cdiv(seqlen_k, bn)
    #    assert head_size == bn
    import sys

    float_max = f32.max_value

    qk_max_elements = 2
    qk_sum_elements = 2

    with hidet.script_module() as script_module:

        @hidet.script
        def func(
            q: f16[batch_size, seqlen_q, num_heads, head_size],
            k: f16[batch_size, seqlen_k, num_heads_k, head_size],
            v: f16[batch_size, seqlen_k, num_heads_k, head_size],
            o: f16[batch_size, seqlen_q, num_heads, head_size],
            lock: i32[batch_size, num_heads, num_pid_m],
            imm: f32[batch_size, num_heads, num_pid_n, seqlen_q],
        ):
            attrs.func_kind = "cuda_kernel"
            attrs.cuda.block_dim = threads
            attrs.cuda.grid_dim = num_pid_m * num_pid_n, bs
            attrs.cuda.dynamic_smem_bytes = dynamic_smem_bytes

            pid = blockIdx.x
            pid_m = pid // num_pid_n
            pid_n = pid % num_pid_n

            bs_idx = blockIdx.y
            batch_idx = bs_idx // num_heads
            head_idx = bs_idx % num_heads

            if pid_n == 0 and threadIdx.x == 0:
                lock[batch_idx, head_idx, pid_m] = 0

            smem_q = dynamic_shared_memory(byte_offset=0, dtype=f16)
            smem_k = dynamic_shared_memory(byte_offset=bm * bk * stages * f16.nbytes, dtype=f16)

            q_regs = register_tensor("float16", shape=[q_elements, 2])
            k_regs = register_tensor("float16", shape=[k_elements, 2])
            qk = register_tensor("float32", shape=[qk_elements])

            for i in grid(qk_elements):
                qk[i] = 0.0

            tg_q = tensor_view(
                q[batch_idx, pid_m * bm :, head_idx, :],
                TensorLayout((bm, head_size), (num_heads * head_size, 1)),
                "global",
            )
            tg_k = tensor_view(
                k[batch_idx, pid_n * bn :, head_idx, :],
                TensorLayout((bn, head_size), (num_heads_k * head_size, 1)),
                "global",
            )

            txgq = partition_src(tg_q, auto_copy())
            txgk = partition_src(tg_k, auto_copy())

            # ts_q = tensor_view(smem_q, layout_auto((bm, head_size)), "shared")
            # ts_k = tensor_view(smem_k, layout_auto((bn, head_size)), "shared")
            ts_q = make_tensor("float16", TensorLayout((bm, bk), (bk, 1)), "shared")
            ts_k = make_tensor("float16", TensorLayout((bn, bk), (bk, 1)), "shared")

            txsq = partition_dst(ts_q, auto_copy())
            txsk = partition_dst(ts_k, auto_copy())

            tr_q = tensor_view(q_regs, layout_auto((bm, inst_h * 2)), "register")
            tr_k = tensor_view(k_regs, layout_auto((bn, inst_h * 2)), "register")
            tr_qk = tensor_view(qk, auto_layout, "register")

            txSq = partition_src(ts_q, auto_copy())
            txrq = partition_dst(tr_q, auto_copy())

            txSk = partition_src(ts_k, auto_copy())
            txrk = partition_dst(tr_k, auto_copy())

            h_tile_max = (bk + inst_h - 1) // inst_h
            for ho in range((head_size + bk - 1) // bk):
                copy(auto_copy((bm, bk)), txgq[:, :, ho], txsq)
                copy(auto_copy((bn, bk)), txgk[:, :, ho], txsk)

                copy(auto_copy(), txSq[:, :, 0], txrq[:, :, 0])
                copy(auto_copy(), txSk[:, :, 0], txrk[:, :, 0])

                for hi in range(h_tile_max):
                    if hi < h_tile_max - 1:
                        copy(auto_copy(), txSq[:, :, hi + 1], txrq[:, :, (hi + 1) % 2])
                        copy(auto_copy(), txSk[:, :, hi + 1], txrk[:, :, (hi + 1) % 2])

                    mma(tiled_mma_qk, tr_qk, txrq[:, :, hi % 2], txrk[:, :, hi % 2], tr_qk)

            lc = ~lock[batch_idx, head_idx, pid_m]
            tr_qk_max = reduce_max(tr_qk, axis=1)

            if pid_n < num_pid_n - 1:
                tg_qk_max = tensor_view(
                    imm[batch_idx, head_idx, pid_n, pid_m * bm :], TensorLayout((bm, bn), (1, 0)), "global"
                )
                txrqk_max = partition_src(tr_qk_max, auto_copy())
                txgqk_max = partition_dst(tg_qk_max, auto_copy())
                copy(auto_copy((bm, bn)), txrqk_max, txgqk_max)
            #    syncthreads()
            #    if threadIdx.x == 0:
            #        atomic_add(lc, 1)
            # else:
            #    qk_max_part = register_tensor("float32", shape=[qk_max_elements])
            #    tr_qk_max_part = tensor_view(qk_max_part, auto_layout, "register")
            #    acquire_seq_semaphore(lc, pid_n)
            #    for i in range(num_pid_n - 1):
            #        tg_qk_max = tensor_view(imm[batch_idx, head_idx, i, pid_m * bm:], TensorLayout((bm, bn), (1, 0)), "global")
            #        txgqk_max = partition_src(tg_qk_max, auto_copy())
            #        txrqk_max = partition_dst(tr_qk_max_part, auto_copy())
            #        copy(auto_copy((bm, bn)), txgqk_max, txrqk_max)
            #        tr_qk_max = elementwise_max(tr_qk_max, tr_qk_max_part)
            #    tg_qk_max = tensor_view(imm[batch_idx, head_idx, pid_n, pid_m * bm:], TensorLayout((bm, bn), (1, 0)), "global")
            #    txrqk_max_final = partition_src(tr_qk_max, auto_copy())
            #    txgqk_max = partition_dst(tg_qk_max, auto_copy())
            #    copy(auto_copy((bm, bn)), txrqk_max_final, txgqk_max)
            #    release_seq_semaphore(lc, pid_n + 1)
            # acquire_seq_semaphore(lc, num_pid_n)
            # tg_qk_max = tensor_view(imm[batch_idx, head_idx, pid_n, pid_m * bm:], TensorLayout((bm, bn), (1, 0)), "global")
            # txgqk_max = partition_src(tg_qk_max, auto_copy())
            # txrqk_max = partition_dst(tr_qk_max, auto_copy())
            # copy(auto_copy((bm, bn)), txgqk_max, txrqk_max)

            tr_qk_exp = exp(tr_qk - tr_qk_max)
            tr_qk_sum = reduce_sum(tr_qk_exp, axis=1)
            if pid_n < num_pid_n - 1:
                tg_qk_sum = tensor_view(
                    imm[batch_idx, head_idx, pid_n, pid_m * bm :], TensorLayout((bm, bn), (1, 0)), "global"
                )
                txrqk_sum = partition_src(tr_qk_sum, auto_copy())
                txgqk_sum = partition_dst(tg_qk_sum, auto_copy())
                copy(auto_copy((bm, bn)), txrqk_sum, txgqk_sum)
            #    syncthreads()
            #    if threadIdx.x == 0:
            #        atomic_sub(lc, 1)
            # else:
            #    qk_sum_part = register_tensor("float32", shape=[qk_sum_elements])
            #    tr_qk_sum_part = tensor_view(qk_sum_part, auto_layout, "register")
            #    acquire_seq_semaphore(lc, 1)
            #    for i in range(num_pid_n - 1):
            #        tg_qk_sum = tensor_view(imm[batch_idx, head_idx, i, pid_m * bm:], TensorLayout((bm, bn), (1, 0)), "global")
            #        txgqk_sum = partition_src(tg_qk_sum, auto_copy())
            #        txrqk_sum = partition_dst(tr_qk_sum_part, auto_copy())
            #        copy(auto_copy((bm, bn)), txgqk_sum, txrqk_sum)
            #        tr_qk_sum = tr_qk_sum + tr_qk_sum_part
            #    tg_qk_sum = tensor_view(imm[batch_idx, head_idx, pid_n, pid_m * bm:], TensorLayout((bm, bn), (1, 0)), "global")
            #    txrqk_sum_final = partition_src(tr_qk_sum, auto_copy())
            #    txgqk_sum = partition_dst(tg_qk_sum, auto_copy())
            #    copy(auto_copy((bm, bn)), txrqk_sum_final, txgqk_sum)
            #    release_seq_semaphore(lc, 0)
            # acquire_seq_semaphore(lc, 0)
            # tg_qk_sum = tensor_view(imm[batch_idx, head_idx, pid_n, pid_m * bm:], TensorLayout((bm, bn), (1, 0)), "global")
            # txgqk_sum = partition_src(tg_qk_sum, auto_copy())
            # txrqk_sum = partition_dst(tr_qk_sum, auto_copy())
            # copy(auto_copy((bm, bn)), txgqk_sum, txrqk_sum)

            tr_qk_o = tr_qk_exp / tr_qk_sum
            tr_qk_f16 = cast(tr_qk_o, f16)

            smem_qk = dynamic_shared_memory(byte_offset=0, dtype=f16)

            ts_qk = tensor_view(smem_qk, TensorLayout((bm, bn), (bn, 1)), "shared")

            # tg_v = tensor_view(
            #    v[batch_idx, pid_n * bn:, head_idx, :], TensorLayout((head_size, bn), (num_heads_k * head_size, 1)), "global"
            # )

            txgq = partition_src(tg_q, auto_copy())

            txrqk1 = partition_src(tr_qk_f16, auto_copy())
            txsqk = partition_dst(ts_qk, auto_copy())
            copy(auto_copy((bm, bn)), txrqk1, txsqk)
            syncthreads()
            qk_regs = register_tensor("float16", shape=[qk1_elements, 2])
            v_regs = register_tensor("float16", shape=[v_elements, 2])
            o_regs = register_tensor("float32", shape=[o_elements])

            for i in grid(o_elements):
                o_regs[i] = 0.0

            smem_v = dynamic_shared_memory(byte_offset=bm * bn * f16.nbytes, dtype=f16)
            ts_v = tensor_view(smem_v, TensorLayout((head_size, bn), (1, head_size)), "shared")
            tr_qk1 = tensor_view(qk_regs, layout_auto((bm, inst_n * 2)), "register")
            tr_v = tensor_view(v_regs, layout_auto((head_size, inst_n * 2)), "register")
            tr_o = tensor_view(o_regs, auto_layout, "register")

            txSqk = partition_src(ts_qk, auto_copy())
            txrqk = partition_dst(tr_qk1, auto_copy())
            txSv = partition_src(ts_v, auto_copy())
            txrv = partition_dst(tr_v, auto_copy())

            copy(auto_copy(), txSqk[:, :, 0], txrqk[:, :, 0])
            copy(auto_copy(), txSv[:, :, 0], txrv[:, :, 0])

            n_tile_max = (bn + inst_n - 1) // inst_n
            for ni in range(n_tile_max):
                if ni < n_tile_max - 1:
                    copy(auto_copy(), txSqk[:, :, ni + 1], txrqk[:, :, (ni + 1) % 2])
                    copy(auto_copy(), txSv[:, :, ni + 1], txrv[:, :, (ni + 1) % 2])
                mma(tiled_mma_o, tr_o, txrqk[:, :, ni % 2], txrv[:, :, ni % 2], tr_o)

            tr_o_f16 = cast(tr_o, f16)
            tr_O = rearrange(tr_o_f16, auto_layout, "register")

            if pid_n == 0:
                tg_o = tensor_view(
                    o[batch_idx, pid_m * bm :, head_idx, 0:],
                    TensorLayout((bm, head_size), (num_heads * head_size, 1)),
                    "global",
                )
                txro = partition_src(tr_O, auto_copy())
                txgo = partition_dst(tg_o, auto_copy())
                copy(auto_copy((bm, head_size)), txro, txgo)
                # release_seq_semaphore(lc, 1)
            else:
                o_part = register_tensor("float16", shape=[o_elements])
                tr_o_part = tensor_view(o_part, auto_layout, "register")
                # acquire_seq_semaphore(lc, pid_n)
                tg_o = tensor_view(
                    o[batch_idx, pid_m * bm :, head_idx, 0:],
                    TensorLayout((bm, head_size), (num_heads * head_size, 1)),
                    "global",
                )
                txgo = partition_src(tg_o, auto_copy())
                txro_part = partition_dst(tr_o_part, auto_copy())
                copy(auto_copy((bm, head_size)), txgo, txro_part)
                tr_O = tr_O + tr_o_part
                txro_final = partition_src(tr_O, auto_copy())
                txgo1 = partition_dst(tg_o, auto_copy())
                copy(auto_copy((bm, head_size)), txro_final, txgo1)
                # release_seq_semaphore(lc, pid_n + 1)

    func = script_module.build()
    # a_mem = hidet.empty([m, k], device="cuda")
    # b_mem = hidet.empty([k, n], device="cuda")
    # c_mem = hidet.empty([m, n], device="cuda")
    # func(a_mem, b_mem, c_mem)
    return func


def flash_decoding_v3(
    batch_size, seqlen_q, num_heads, head_size, seqlen_k, num_heads_k, num_seqk_parallel_parts: int = 128
):
    from hidet.lang.types import u32, i32, f16, f32
    from hidet.lang import attrs
    from hidet.lang import shared_tensor, register_tensor
    from hidet.lang.cuda import syncthreads, cp_async_wait_all
    from hidet.lang.cuda import blockIdx, threadIdx, dynamic_shared_memory
    from hidet.lang.cuda import cp_async, cp_async_commit_group, cp_async_wait_group

    # bm * bn
    a = TensorLayout(((4, 8), (2, 2)), ((16, 1), (8, 64)))
    b = TensorLayout(((4, 8), (2, 2, 2)), ((32, 1), (16, 8, 128)))
    c = TensorLayout(((4, 8), (2, 2)), ((2, 8), (1, 64)))
    mma_atom = MmaAtom("warp", (8, 16, 16), a, b, c, c, (1, 1))
    warp_in_threadblock = Level("warp", "thread_block", (1, 4), TensorLayout((1, 4)), (1, 2))
    tiled_mma_qk = TiledMma(mma_atom, [warp_in_threadblock])

    print(tiled_mma_qk.str_indented())
    q_shape, q_tv_layout = tiled_mma_qk.a_tv_layout()
    k_shape, k_tv_layout = tiled_mma_qk.b_tv_layout()
    qk_shape, qk_tv_layout = tiled_mma_qk.c_tv_layout()
    print(q_shape, q_tv_layout)
    print(k_shape, k_tv_layout)
    print(qk_shape, qk_tv_layout)

    _, q_v = canonicalize(q_tv_layout)
    _, k_v = canonicalize(k_tv_layout)
    qk_t, qk_v = canonicalize(qk_tv_layout)

    threads = qk_t.size()

    bm, inst_h = q_shape
    bn, inst_h_ = k_shape
    bm_, bn_ = qk_shape
    assert bm == bm_ and bn == bn_ and inst_h == inst_h_

    a = TensorLayout(((4, 8), (2, 2)), ((16, 1), (8, 64)))
    b = TensorLayout(((4, 8), (2, 2, 2)), ((32, 1), (16, 8, 128)))
    c = TensorLayout(((4, 8), (2, 2)), ((2, 8), (1, 64)))
    mma_atom = MmaAtom("warp", (8, 16, 16), a, b, c, c, (1, 1))
    warp_in_threadblock = Level("warp", "thread_block", (1, 4), TensorLayout((1, 4)), (1, 2))
    tiled_mma_o = TiledMma(mma_atom, [warp_in_threadblock])

    print(tiled_mma_o.str_indented())
    qk_shape, qk_tv_layout = tiled_mma_o.a_tv_layout()
    v_shape, v_tv_layout = tiled_mma_o.b_tv_layout()
    o_shape, o_tv_layout = tiled_mma_o.c_tv_layout()
    print(qk_shape, qk_tv_layout)
    print(v_shape, v_tv_layout)
    print(o_shape, o_tv_layout)

    _, inst_n = qk_shape

    _, qk_v = canonicalize(qk_tv_layout)
    _, v_v = canonicalize(v_tv_layout)
    _, o_v = canonicalize(o_tv_layout)

    from hidet.ir.cute.algorithm import auto_copy, auto_mma
    from hidet.ir.cute import auto_layout, layout_auto

    dynamic_smem_bytes = (bm * head_size + bn * head_size + bm * bn + bn * head_size) * f16.nbytes

    from hidet.utils.py import cdiv
    from hidet.lang import grid

    n_size = seqlen_q * num_heads
    bs = batch_size * num_heads
    assert num_heads == num_heads_k
    #    assert head_size == bn
    import sys

    float_max = f32.max_value

    assert seqlen_k % num_seqk_parallel_parts == 0
    seqk_partition = seqlen_k // num_seqk_parallel_parts
    assert seqk_partition % bn == 0
    num_pid_m = cdiv(seqlen_q, bm)

    with hidet.script_module() as script_module:

        @hidet.script
        def func(
            q: f16[batch_size, seqlen_q, num_heads, head_size],
            k: f16[batch_size, seqlen_k, num_heads_k, head_size],
            v: f16[batch_size, seqlen_k, num_heads_k, head_size],
            o: f16[batch_size, seqlen_q, num_heads, head_size],
            lock: i32[batch_size, num_heads, num_pid_m],
            li: f32[batch_size, num_heads, num_seqk_parallel_parts, seqlen_q],
            mi: f32[batch_size, num_heads, num_seqk_parallel_parts, seqlen_q],
            oi: f16[batch_size, seqlen_q, num_heads, num_seqk_parallel_parts, head_size],
        ):
            attrs.func_kind = "cuda_kernel"
            attrs.cuda.block_dim = threads
            attrs.cuda.grid_dim = bs * num_pid_m * num_seqk_parallel_parts
            attrs.cuda.dynamic_smem_bytes = dynamic_smem_bytes

            pid = blockIdx.x
            seqk_part = pid % num_seqk_parallel_parts
            pid = pid // num_seqk_parallel_parts
            pid_m = pid % num_pid_m
            pid = pid // num_pid_m
            batch_idx = pid // num_heads
            head_idx = pid % num_heads

            if seqk_part == 0 and threadIdx.x == 0:
                lock[batch_idx, head_idx, pid_m] = 0

            tg_q = tensor_view(
                q[batch_idx, pid_m * bm :, head_idx, :],
                TensorLayout((bm, head_size), (num_heads * head_size, 1)),
                "global",
            )
            txgq = partition_src(tg_q, auto_copy())
            tg_k = tensor_view(
                k[batch_idx, seqk_part * seqk_partition :, head_idx, :],
                TensorLayout((bn, head_size), (num_heads_k * head_size, 1)),
                "global",
            )
            txgk = partition_src(tg_k, auto_copy())
            tg_v = tensor_view(
                v[batch_idx, seqk_part * seqk_partition :, head_idx, :],
                TensorLayout((head_size, seqlen_k), (1, num_heads_k * head_size)),
                "global",
            )
            txgv = partition_src(tg_v, auto_copy())

            ts_q = make_tensor("float16", TensorLayout((bm, head_size), (head_size, 1)), "shared")
            txsq = partition_dst(ts_q, auto_copy())
            ts_k = make_tensor("float16", TensorLayout((bn, head_size), (head_size, 1)), "shared")
            txsk = partition_dst(ts_k, auto_copy())
            ts_v = make_tensor("float16", TensorLayout((head_size, bn), (1, head_size)), "shared")
            txsv = partition_dst(ts_v, auto_copy())
            ts_qk = make_tensor("float16", layout_auto((bm, bn)), "shared")
            txsqk = partition_dst(ts_qk, auto_copy())

            copy(auto_copy((bm, head_size)), txgq, txsq)
            copy(auto_copy((bn, head_size)), txgk, txsk)
            cp_async_commit_group()

            tr_q = make_tensor("float16", layout_auto((bm, inst_h * 2)), "register")
            tr_k = make_tensor("float16", layout_auto((bn, inst_h * 2)), "register")
            tr_v = make_tensor("float16", layout_auto((head_size, inst_n * 2)), "register")
            tr_qk = make_tensor("float32", auto_layout, "register")
            tr_qk1 = make_tensor("float16", layout_auto((bm, inst_n * 2)), "register")
            tr_o = make_tensor("float32", auto_layout, "register")
            tr_qk_max = make_tensor("float32", layout_auto((bm, bn), (1, 0)), "register")
            tr_qk_sum = make_tensor("float32", layout_auto((bm, bn), (1, 0)), "register")

            fill(tr_qk_max, -float_max)
            fill(tr_qk_sum, 0.0)
            fill(tr_o, 0.0)

            txSq = partition_src(ts_q, auto_copy())
            txrq = partition_dst(tr_q, auto_copy())

            txSk = partition_src(ts_k, auto_copy())
            txrk = partition_dst(tr_k, auto_copy())

            txSv = partition_src(ts_v, auto_copy())
            txrv = partition_dst(tr_v, auto_copy())

            txSqk = partition_src(ts_qk, auto_copy())
            txrqk = partition_dst(tr_qk1, auto_copy())

            cp_async_wait_group(0)
            syncthreads()
            copy(auto_copy(), txSq[:, :, 0], txrq[:, :, 0])
            copy(auto_copy(), txSk[:, :, 0], txrk[:, :, 0])

            h_tile_max = (head_size + inst_h - 1) // inst_h
            n_tile_max = (bn + inst_n - 1) // inst_n
            no_size = (seqk_partition + bn - 1) // bn
            for no in range(no_size):
                fill(tr_qk, 0.0)

                cp_async_wait_group(0)
                syncthreads()

                if no >= 1:
                    copy(auto_copy(), txSk[:, :, 0], txrk[:, :, 0])

                copy(auto_copy((head_size, bn)), txgv[:, :, no], txsv)
                cp_async_commit_group()

                for hi in grid(h_tile_max, attrs="u"):
                    h_tile_next = (hi + 1) % h_tile_max
                    copy(auto_copy(), txSq[:, :, h_tile_next], txrq[:, :, (hi + 1) % 2])
                    if hi < h_tile_max - 1:
                        copy(auto_copy(), txSk[:, :, h_tile_next], txrk[:, :, (hi + 1) % 2])
                    mma(tiled_mma_qk, tr_qk, txrq[:, :, hi % 2], txrk[:, :, hi % 2], tr_qk)

                tr_qk1_max = reduce_max(tr_qk, axis=1)
                scale = exp(tr_qk_max - elementwise_max(tr_qk1_max, tr_qk_max))
                tr_qk_max = elementwise_max(tr_qk1_max, tr_qk_max)
                tr_qk_exp = exp(tr_qk - tr_qk_max)
                tr_qk1_sum = reduce_sum(tr_qk_exp, axis=1)
                tr_qk_sum = tr_qk_sum * scale
                alpha = tr_qk_sum / (tr_qk_sum + tr_qk1_sum)
                if no >= 1:
                    alpha1 = broadcast_to(alpha, tr_o)
                    tr_o = tr_o * alpha1
                tr_qk_sum = tr_qk_sum + tr_qk1_sum
                tr_qk_o = tr_qk_exp / tr_qk_sum
                tr_qk_f16 = cast(tr_qk_o, f16)
                syncthreads()
                txrqk1 = partition_src(tr_qk_f16, auto_copy())
                copy(auto_copy((bm, bn)), txrqk1, txsqk)
                cp_async_wait_group(0)
                syncthreads()

                if no < no_size - 1:
                    tg_k1 = tensor_view(
                        k[batch_idx, seqk_part * seqk_partition + (no + 1) * bn :, head_idx, :,],
                        TensorLayout((bn, head_size), (num_heads_k * head_size, 1)),
                        "global",
                    )
                    txgk1 = partition_src(tg_k1, auto_copy())
                    copy(auto_copy((bn, head_size)), txgk1[:, :], txsk[:, :])
                cp_async_commit_group()

                copy(auto_copy(), txSqk[:, :, 0], txrqk[:, :, 0])
                copy(auto_copy(), txSv[:, :, 0], txrv[:, :, 0])

                for ni in grid(n_tile_max, attrs="u"):
                    if ni < n_tile_max - 1:
                        copy(auto_copy(), txSqk[:, :, ni + 1], txrqk[:, :, (ni + 1) % 2])
                        copy(auto_copy(), txSv[:, :, ni + 1], txrv[:, :, (ni + 1) % 2])
                    mma(tiled_mma_o, tr_o, txrqk[:, :, ni % 2], txrv[:, :, ni % 2], tr_o)

            lc = ~lock[batch_idx, head_idx, pid_m]

            if seqk_part > 0:
                tr_o_f16 = cast(tr_o, f16)
                tr_O = rearrange(tr_o_f16, auto_layout, "register")
                tg_oi = tensor_view(
                    oi[batch_idx, pid_m * bm :, head_idx, seqk_part, 0:],
                    TensorLayout((bm, head_size), (num_heads * num_seqk_parallel_parts * head_size, 1)),
                    "global",
                )
                tg_li = tensor_view(
                    li[batch_idx, head_idx, seqk_part, pid_m * bm :], TensorLayout((bm, bn), (1, 0)), "global"
                )
                tg_mi = tensor_view(
                    mi[batch_idx, head_idx, seqk_part, pid_m * bm :], TensorLayout((bm, bn), (1, 0)), "global"
                )
                txrx_o = partition_src(tr_O, auto_copy())
                txgx_oi = partition_dst(tg_oi, auto_copy())
                copy(auto_copy((bm, head_size)), txrx_o, txgx_oi)
                txrqk_max = partition_src(tr_qk_max, auto_copy())
                txgqk_max = partition_dst(tg_li, auto_copy())
                copy(auto_copy((bm, bn)), txrqk_max, txgqk_max)
                txrqk_sum = partition_src(tr_qk_sum, auto_copy())
                txgqk_sum = partition_dst(tg_mi, auto_copy())
                copy(auto_copy((bm, bn)), txrqk_sum, txgqk_sum)
                syncthreads()
                if threadIdx.x == 0:
                    atomic_add(lc, 1)
            else:
                tr_oi = make_tensor("float16", auto_layout, "register")
                tr_li = make_tensor("float32", layout_auto((bm, bn), (1, 0)), "register")
                tr_mi = make_tensor("float32", layout_auto((bm, bn), (1, 0)), "register")
                acquire_seq_semaphore(lc, num_seqk_parallel_parts - 1)
                for i in range(num_seqk_parallel_parts - 1):
                    tg_oi = tensor_view(
                        oi[batch_idx, pid_m * bm :, head_idx, i + 1, 0:],
                        TensorLayout((bm, head_size), (num_heads * num_seqk_parallel_parts * head_size, 1)),
                        "global",
                    )
                    tg_li = tensor_view(
                        li[batch_idx, head_idx, i + 1, pid_m * bm :], TensorLayout((bm, bn), (1, 0)), "global"
                    )
                    tg_mi = tensor_view(
                        mi[batch_idx, head_idx, i + 1, pid_m * bm :], TensorLayout((bm, bn), (1, 0)), "global"
                    )
                    txgx_oi = partition_dst(tg_oi, auto_copy())
                    txrx_oi = partition_src(tr_oi, auto_copy())
                    copy(auto_copy((bm, head_size)), txgx_oi, txrx_oi)
                    txgli = partition_src(tg_li, auto_copy())
                    txrli = partition_dst(tr_li, auto_copy())
                    copy(auto_copy((bm, bn)), txgli, txrli)
                    txgmi = partition_src(tg_mi, auto_copy())
                    txrmi = partition_dst(tr_mi, auto_copy())
                    copy(auto_copy((bm, bn)), txgmi, txrmi)
                    scale = exp(tr_qk_max - elementwise_max(tr_qk_max, tr_li))
                    tr_qk_max = elementwise_max(tr_qk_max, tr_li)
                    scale1 = exp(tr_li - tr_qk_max)
                    tr_qk_sum = tr_qk_sum * scale
                    tr_mi = tr_mi * scale1
                    alpha = tr_qk_sum / (tr_qk_sum + tr_mi)
                    tr_qk_sum = tr_qk_sum + tr_mi
                    alpha1 = broadcast_to(alpha, tr_o)
                    tr_o = tr_o * alpha1 + cast(tr_oi, f32) * (1.0 - alpha1)
                tr_o_f16 = cast(tr_o, f16)
                tr_O = rearrange(tr_o_f16, auto_layout, "register")

                tg_o = tensor_view(
                    o[batch_idx, pid_m * bm :, head_idx, 0:],
                    TensorLayout((bm, head_size), (num_heads * head_size, 1)),
                    "global",
                )
                txrx_o = partition_src(tr_O, auto_copy())
                txgx_o = partition_dst(tg_o, auto_copy())
                copy(auto_copy((bm, head_size)), txrx_o, txgx_o)

    func = script_module.build()
    # a_mem = hidet.empty([m, k], device="cuda")
    # b_mem = hidet.empty([k, n], device="cuda")
    # c_mem = hidet.empty([m, n], device="cuda")
    # func(a_mem, b_mem, c_mem)
    return func


def data(
    batch_size,
    seqlen_q,
    num_heads,
    head_size,
    seqlen_k,
    num_heads_k,
    dtype="float16",
    device="cuda",
    return_hidet=False,
):
    dtype = getattr(torch, dtype)
    lo = -3
    hi = 3
    q = torch.randint(low=lo, high=hi, size=(batch_size, seqlen_q, num_heads, head_size), dtype=dtype, device=device)
    k = torch.randint(low=lo, high=hi, size=(batch_size, seqlen_k, num_heads_k, head_size), dtype=dtype, device=device)
    v = torch.randint(low=lo, high=hi, size=(batch_size, seqlen_k, num_heads_k, head_size), dtype=dtype, device=device)
    o = torch.empty((batch_size, seqlen_q, num_heads, head_size), dtype=dtype, device=device)

    q = q
    k = k / head_size
    v = v
    if return_hidet:
        q = hidet.from_torch(q)
        k = hidet.from_torch(k)
        v = hidet.from_torch(v)
        o = hidet.from_torch(o)

    return q, k, v, o


@pytest.mark.parametrize(
    "batch_size,num_heads,num_heads_k,head_size,seqlen_q,seqlen_k",
    [(1, 32, 32, 128, 8, 4096), (1, 32, 32, 128, 8, 2048), (1, 32, 32, 128, 8, 1024)],
)
def test_v1(batch_size, num_heads, num_heads_k, head_size, seqlen_q, seqlen_k):
    flash_decoding_v1(batch_size, seqlen_q, num_heads, head_size, seqlen_k, num_heads_k)


@pytest.mark.parametrize(
    "batch_size,num_heads,num_heads_k,head_size,seqlen_q,seqlen_k",
    [(1, 32, 32, 128, 8, 4096), (1, 32, 32, 128, 8, 2048), (1, 32, 32, 128, 8, 1024)],
)
def test_flash_decoding(batch_size, num_heads, num_heads_k, head_size, seqlen_q, seqlen_k):
    num_seqk_parallel_parts = 2
    func = flash_decoding_v3(batch_size, seqlen_q, num_heads, head_size, seqlen_k, num_heads_k, num_seqk_parallel_parts)
    q, k, v, o = data(batch_size, seqlen_q, num_heads, head_size, seqlen_k, num_heads_k, return_hidet=True)

    def fn():
        bm = 8
        num_pid_m = cdiv(seqlen_q, bm)
        lock = torch.empty((batch_size, num_heads, num_pid_m), dtype=torch.int32, device="cuda")
        li = torch.empty((batch_size, num_heads, num_seqk_parallel_parts, seqlen_q), dtype=torch.float32, device="cuda")
        mi = torch.empty((batch_size, num_heads, num_seqk_parallel_parts, seqlen_q), dtype=torch.float32, device="cuda")
        oi = torch.empty(
            (batch_size, seqlen_q, num_heads, num_seqk_parallel_parts, head_size), dtype=torch.float16, device="cuda"
        )
        func(q, k, v, o, lock, li, mi, oi)

    mean, min_lat, max_lat = bench(fn, ())
    flops = 2.0 * (
        batch_size * seqlen_q * num_heads * seqlen_k * head_size
        + batch_size * seqlen_q * num_heads_k * seqlen_k * head_size
    )
    from hidet.ir.dtypes import f16

    memory = f16.nbytes * (
        batch_size * num_heads * seqlen_q * head_size
        + batch_size * num_heads_k * seqlen_k * head_size
        + batch_size * num_heads_k * seqlen_k * head_size
        + batch_size * seqlen_q * num_heads * head_size
    )
    print("time={:.3f} ms, performance={:.3f} TFLOPS".format(mean, flops / (1e9 * mean)))
    print("time={:.3f} ms, bandwidth={:.3f} GB/s".format(mean, memory / (1e6 * mean)))

    from hidet.utils.benchmark import do_bench

    mean = do_bench(fn, percentiles=None)
    print("time={:.3f} ms, performance={:.3f} TFLOPS".format(mean, flops / (1e9 * mean)))
    print("time={:.3f} ms, bandwidth={:.3f} GB/s".format(mean, memory / (1e6 * mean)))

    q = q.torch()
    k = k.torch()
    v = v.torch()
    o = o.torch()

    softmax = torch.nn.Softmax(dim=1)

    def fn():
        q1 = q.permute(0, 2, 1, 3)
        q2 = q1.view(batch_size * num_heads, seqlen_q, head_size)
        k1 = k.permute(0, 2, 3, 1)
        k2 = k1.view(batch_size * num_heads_k, head_size, seqlen_k)
        qk = q2 @ k2
        v1 = v.permute(0, 2, 1, 3)
        v2 = v1.view(batch_size * num_heads_k, seqlen_k, head_size)
        qk = qk.view(batch_size * num_heads * seqlen_q, seqlen_k)
        qk = softmax(qk.to(torch.float32))
        qk = qk.to(torch.float16)
        qk = qk.view(batch_size * num_heads, seqlen_q, seqlen_k)
        o1 = qk @ v2
        o2 = o1.view(batch_size, num_heads, seqlen_q, head_size)
        o3 = o2.permute(0, 2, 1, 3)
        return o3.contiguous()

    from hidet.utils.benchmark import do_bench

    mean = do_bench(fn, percentiles=None)
    print("time={:.3f} ms, performance={:.3f} TFLOPS".format(mean, flops / (1e9 * mean)))

    o2 = fn()
    import numpy as np

    np.set_printoptions(threshold=3000, linewidth=200, edgeitems=100)
    np.testing.assert_allclose(actual=o.cpu().numpy(), desired=o2.cpu().numpy(), rtol=1e-2)
