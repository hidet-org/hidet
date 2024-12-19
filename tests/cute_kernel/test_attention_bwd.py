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
    transpose,
    cute_atomic_add,
)
from hidet.lang.mapping import auto_map
from hidet.ir.primitives.cuda.mutex import release_seq_semaphore, acquire_seq_semaphore
from hidet.ir.primitives.cuda.atomic import atomic_add, atomic_sub
from hidet.utils.py import cdiv

from quant_utils import bench, canonicalize


def get_tiled_mma(head_size):
    """
    Hardcoded tiled_mma for RTX4090
    TODO: how to automatically determine these tile_mma?
    """
    if head_size == 128:
        # bc = 64
        # br = 64
        # head_size = 128
        a = TensorLayout(((4, 8), (2, 2, 2)), ((32, 1), (16, 8, 128)))
        b = TensorLayout(((4, 8), (2, 2)), ((16, 1), (8, 64)))
        c = TensorLayout(((4, 8), (2, 2)), ((32, 1), (16, 8)))
        mma_atom = MmaAtom("warp", (16, 8, 16), a, b, c, c, (1, 2))
        warp_in_threadblock = Level("warp", "thread_block", (4, 2), TensorLayout((4, 2)), (1, 2))
        tiled_mma_qk = TiledMma(mma_atom, [warp_in_threadblock])

        # bc = 64
        # head_size = 128
        a = TensorLayout(((4, 8), (2, 2, 2)), ((32, 1), (16, 8, 128)))
        b = TensorLayout(((4, 8), (2, 2)), ((16, 1), (8, 64)))
        c = TensorLayout(((4, 8), (2, 2)), ((32, 1), (16, 8)))
        mma_atom = MmaAtom("warp", (16, 8, 16), a, b, c, c, (1, 2))
        warp_in_threadblock = Level("warp", "thread_block", (4, 2), TensorLayout((4, 2)), (1, 4))
        tiled_mma_pdo = TiledMma(mma_atom, [warp_in_threadblock])

        # bc = 64
        # head_size = 128
        a = TensorLayout(((4, 8), (2, 2, 2)), ((32, 1), (16, 8, 128)))
        b = TensorLayout(((4, 8), (2, 2)), ((16, 1), (8, 64)))
        c = TensorLayout(((4, 8), (2, 2)), ((32, 1), (16, 8)))
        mma_atom = MmaAtom("warp", (16, 8, 16), a, b, c, c, (1, 2))
        warp_in_threadblock = Level("warp", "thread_block", (4, 2), TensorLayout((4, 2)), (1, 4))
        tiled_mma_dsq = TiledMma(mma_atom, [warp_in_threadblock])

        # br = 64
        # head_size = 128
        a = TensorLayout(((4, 8), (2, 2, 2)), ((32, 1), (16, 8, 128)))
        b = TensorLayout(((4, 8), (2, 2)), ((16, 1), (8, 64)))
        c = TensorLayout(((4, 8), (2, 2)), ((32, 1), (16, 8)))
        mma_atom = MmaAtom("warp", (16, 8, 16), a, b, c, c, (1, 2))
        warp_in_threadblock = Level("warp", "thread_block", (4, 2), TensorLayout((4, 2)), (1, 4))
        tiled_mma_dsk = TiledMma(mma_atom, [warp_in_threadblock])

        return tiled_mma_qk, tiled_mma_pdo, tiled_mma_dsq, tiled_mma_dsk
    elif head_size == 64:
        # bc = 128
        # br = 64
        # head_size = 64
        a = TensorLayout(((4, 8), (2, 2, 2)), ((32, 1), (16, 8, 128)))
        b = TensorLayout(((4, 8), (2, 2)), ((16, 1), (8, 64)))
        c = TensorLayout(((4, 8), (2, 2)), ((32, 1), (16, 8)))
        mma_atom = MmaAtom("warp", (16, 8, 16), a, b, c, c, (1, 2))
        warp_in_threadblock = Level("warp", "thread_block", (4, 2), TensorLayout((4, 2)), (2, 2))
        tiled_mma_qk = TiledMma(mma_atom, [warp_in_threadblock])

        # bc = 128
        # head_size = 64
        a = TensorLayout(((4, 8), (2, 2, 2)), ((32, 1), (16, 8, 128)))
        b = TensorLayout(((4, 8), (2, 2)), ((16, 1), (8, 64)))
        c = TensorLayout(((4, 8), (2, 2)), ((32, 1), (16, 8)))
        mma_atom = MmaAtom("warp", (16, 8, 16), a, b, c, c, (1, 2))
        warp_in_threadblock = Level("warp", "thread_block", (4, 2), TensorLayout((4, 2)), (2, 2))
        tiled_mma_pdo = TiledMma(mma_atom, [warp_in_threadblock])

        # bc = 128
        # head_size = 64
        a = TensorLayout(((4, 8), (2, 2, 2)), ((32, 1), (16, 8, 128)))
        b = TensorLayout(((4, 8), (2, 2)), ((16, 1), (8, 64)))
        c = TensorLayout(((4, 8), (2, 2)), ((32, 1), (16, 8)))
        mma_atom = MmaAtom("warp", (16, 8, 16), a, b, c, c, (1, 2))
        warp_in_threadblock = Level("warp", "thread_block", (4, 2), TensorLayout((4, 2)), (2, 2))
        tiled_mma_dsq = TiledMma(mma_atom, [warp_in_threadblock])

        # br = 64
        # head_size = 64
        a = TensorLayout(((4, 8), (2, 2, 2)), ((32, 1), (16, 8, 128)))
        b = TensorLayout(((4, 8), (2, 2)), ((16, 1), (8, 64)))
        c = TensorLayout(((4, 8), (2, 2)), ((32, 1), (16, 8)))
        mma_atom = MmaAtom("warp", (16, 8, 16), a, b, c, c, (1, 2))
        warp_in_threadblock = Level("warp", "thread_block", (2, 4), TensorLayout((2, 4)), (2, 1))
        tiled_mma_dsk = TiledMma(mma_atom, [warp_in_threadblock])

        return tiled_mma_qk, tiled_mma_pdo, tiled_mma_dsq, tiled_mma_dsk
    else:
        raise NotImplementedError()


def flash_attention_v2_bwd(batch_size, seqlen_q, num_heads, head_size, seqlen_k, num_heads_k):
    from hidet.lang.types import u32, i32, f16, f32
    from hidet.lang import attrs
    from hidet.lang import shared_tensor, register_tensor
    from hidet.lang.cuda import syncthreads, cp_async_wait_all
    from hidet.lang.cuda import blockIdx, threadIdx, dynamic_shared_memory
    from hidet.lang.cuda import cp_async, cp_async_commit_group, cp_async_wait_group

    # bm * bn
    a = TensorLayout(((4, 8), (2, 2, 2)), ((32, 1), (16, 8, 128)))
    b = TensorLayout(((4, 8), (2, 2)), ((16, 1), (8, 64)))
    c = TensorLayout(((4, 8), (2, 2)), ((32, 1), (16, 8)))
    mma_atom = MmaAtom("warp", (16, 8, 16), a, b, c, c, (1, 2))
    warp_in_threadblock = Level("warp", "thread_block", (4, 1), TensorLayout((4, 1)), (1, 2))
    tiled_mma_qk = TiledMma(mma_atom, [warp_in_threadblock])

    k_shape, k_tv_layout = tiled_mma_qk.a_tv_layout()
    q_shape, q_tv_layout = tiled_mma_qk.b_tv_layout()
    qk_shape, qk_tv_layout = tiled_mma_qk.c_tv_layout()
    print(tiled_mma_qk.str_indented())

    qk_t, qk_v = canonicalize(qk_tv_layout)

    threads = qk_t.size()
    br, inst_h = q_shape
    bc, inst_h_ = k_shape
    assert inst_h == inst_h_

    a = TensorLayout(((4, 8), (2, 2, 2)), ((32, 1), (16, 8, 128)))
    b = TensorLayout(((4, 8), (2, 2)), ((16, 1), (8, 64)))
    c = TensorLayout(((4, 8), (2, 2)), ((32, 1), (16, 8)))
    mma_atom = MmaAtom("warp", (16, 8, 16), a, b, c, c, (1, 2))
    warp_in_threadblock = Level("warp", "thread_block", (4, 1), TensorLayout((4, 1)), (1, 8))
    tiled_mma_pdo = TiledMma(mma_atom, [warp_in_threadblock])

    print(tiled_mma_pdo.str_indented())
    p_shape, p_tv_layout = tiled_mma_pdo.a_tv_layout()
    do_shape, do_tv_layout = tiled_mma_pdo.b_tv_layout()
    dv_shape, dv_tv_layout = tiled_mma_pdo.c_tv_layout()
    print(p_shape, p_tv_layout)
    print(do_shape, do_tv_layout)
    print(dv_shape, dv_tv_layout)

    _, inst_r = p_shape

    # bc = 64
    # br = 32
    # dsk = br x head_size
    a = TensorLayout(((4, 8), (2, 2, 2)), ((32, 1), (16, 8, 128)))
    b = TensorLayout(((4, 8), (2, 2)), ((16, 1), (8, 64)))
    c = TensorLayout(((4, 8), (2, 2)), ((32, 1), (16, 8)))
    mma_atom = MmaAtom("warp", (16, 8, 16), a, b, c, c, (1, 2))
    warp_in_threadblock = Level("warp", "thread_block", (2, 2), TensorLayout((2, 2)), (1, 4))
    tiled_mma_dsk = TiledMma(mma_atom, [warp_in_threadblock])

    print(tiled_mma_dsk.str_indented())
    dst_shape, dst_tv_layout = tiled_mma_dsk.a_tv_layout()
    kt_shape, kt_tv_layout = tiled_mma_dsk.b_tv_layout()
    dq_shape, dq_tv_layout = tiled_mma_dsk.c_tv_layout()
    print(dst_shape, dst_tv_layout)
    print(kt_shape, kt_tv_layout)
    print(dq_shape, dq_tv_layout)

    _, inst_c = dst_shape

    # dsq = bc x head_size
    a = TensorLayout(((4, 8), (2, 2, 2)), ((32, 1), (16, 8, 128)))
    b = TensorLayout(((4, 8), (2, 2)), ((16, 1), (8, 64)))
    c = TensorLayout(((4, 8), (2, 2)), ((32, 1), (16, 8)))
    mma_atom = MmaAtom("warp", (16, 8, 16), a, b, c, c, (1, 2))
    warp_in_threadblock = Level("warp", "thread_block", (2, 2), TensorLayout((2, 2)), (2, 4))
    tiled_mma_dsq = TiledMma(mma_atom, [warp_in_threadblock])

    print(tiled_mma_dsq.str_indented())
    ds_shape, ds_tv_layout = tiled_mma_dsq.a_tv_layout()
    qt_shape, qt_tv_layout = tiled_mma_dsq.b_tv_layout()
    dk_shape, dk_tv_layout = tiled_mma_dsq.c_tv_layout()
    print(ds_shape, ds_tv_layout)
    print(qt_shape, qt_tv_layout)
    print(dk_shape, dk_tv_layout)

    from hidet.ir.cute.algorithm import auto_copy, auto_mma
    from hidet.ir.cute import auto_layout, layout_auto

    from hidet.lang import grid

    n_size = seqlen_q * num_heads
    bs = batch_size * num_heads
    assert num_heads == num_heads_k
    #    assert head_size == bn
    import sys

    float_max = f32.max_value
    num_parallel_seqk_parts = 8
    seqk_partition = bc
    while seqk_partition * num_parallel_seqk_parts < seqlen_k:
        seqk_partition += seqk_partition

    tr = cdiv(seqlen_q, br)
    tc = cdiv(seqk_partition, bc)
    print(bc, br)

    with hidet.script_module() as script_module:

        @hidet.script
        def func(
            q: f16[batch_size, seqlen_q, num_heads, head_size],
            k: f16[batch_size, seqlen_k, num_heads_k, head_size],
            v: f16[batch_size, seqlen_k, num_heads_k, head_size],
            o: f16[batch_size, seqlen_q, num_heads, head_size],
            do_1: f16[batch_size, seqlen_q, num_heads, head_size],
            dq: f16[batch_size, seqlen_q, num_heads, head_size],
            dk: f16[batch_size, seqlen_k, num_heads_k, head_size],
            dv: f16[batch_size, seqlen_k, num_heads_k, head_size],
            l: f32[batch_size, num_heads, seqlen_q],
            m: f32[batch_size, num_heads, seqlen_q],
            dq_seqk_parallel_parts: f16[batch_size, seqlen_q, num_heads, num_parallel_seqk_parts, head_size],
            lock: i32[batch_size, head_size, num_parallel_seqk_parts],
        ):
            attrs.func_kind = "cuda_kernel"
            attrs.cuda.block_dim = threads
            attrs.cuda.grid_dim = num_parallel_seqk_parts, bs
            attrs.cuda.dynamic_smem_bytes = 0

            seqk_part = blockIdx.x
            pid = blockIdx.y
            batch_idx = pid // num_heads
            head_idx = pid % num_heads

            # Q -> br x head_size
            # K -> bc x head_size
            # S -> bc x br
            # P -> bc x br
            # dV -> bc x head_size
            # dO -> head_size x br
            # O -> head_size x br
            # dP -> bc x br
            # dOt -> br x head_size
            # V -> bc x head_size
            # dS -> bc x br
            # Qt -> head_size x br
            # dSt -> br x bc
            # Kt -> head_size x bc

            h_tile_max = (head_size + inst_h - 1) // inst_h
            r_tile_max = (br + inst_r - 1) // inst_r
            c_tile_max = (bc + inst_c - 1) // inst_c
            tr_q = make_tensor("float16", layout_auto((br, inst_h * 2)), "register")
            tr_k = make_tensor("float16", layout_auto((bc, inst_h * 2)), "register")
            tr_v = make_tensor("float16", layout_auto((bc, inst_r * 2)), "register")
            tr_dv = make_tensor("float32", auto_layout, "register")
            tr_dk = make_tensor("float32", auto_layout, "register")

            ts_q = make_tensor("float16", TensorLayout((br, head_size), (head_size, 1)), "shared")
            txSq = partition_src(ts_q, auto_copy())
            txsq = partition_dst(ts_q, auto_copy())

            ts_k = make_tensor("float16", TensorLayout((bc, head_size), (head_size, 1)), "shared")
            txSk = partition_src(ts_k, auto_copy())
            txsk = partition_dst(ts_k, auto_copy())

            ts_v = make_tensor("float16", TensorLayout((bc, head_size), (head_size, 1)), "shared")
            txSv = partition_src(ts_v, auto_copy())
            txsv = partition_dst(ts_v, auto_copy())

            ts_do = make_tensor("float16", TensorLayout((head_size, br), (1, head_size)), "shared")
            txSdo = partition_src(ts_do, auto_copy())
            txsdo = partition_dst(ts_do, auto_copy())

            ts_dot = transpose(ts_do, 1, 0)
            txSdot = partition_src(ts_dot, auto_copy())

            ts_ds = make_tensor("float16", TensorLayout((bc, br), (br, 1)), "shared")
            txSds = partition_src(ts_ds, auto_copy())
            txsds = partition_dst(ts_ds, auto_copy())

            ts_dst = transpose(ts_ds, 1, 0)
            txSdst = partition_src(ts_dst, auto_copy())

            ts_qt = transpose(ts_q, 1, 0)
            txSqt = partition_src(ts_qt, auto_copy())

            ts_kt = transpose(ts_k, 1, 0)
            txSkt = partition_src(ts_kt, auto_copy())

            for j in grid(tc):
                fill(tr_dv, 0.0)
                fill(tr_dk, 0.0)

                tg_k = tensor_view(
                    k[batch_idx, seqk_part * seqk_partition + j * bc :, head_idx, :],
                    TensorLayout((bc, head_size), (num_heads_k * head_size, 1)),
                    "global",
                )
                txgk = partition_src(tg_k, auto_copy())
                tg_v = tensor_view(
                    v[batch_idx, seqk_part * seqk_partition + j * bc :, head_idx, :],
                    TensorLayout((bc, head_size), (num_heads_k * head_size, 1)),
                    "global",
                )
                txgv = partition_src(tg_v, auto_copy())

                tg_q = tensor_view(
                    q[batch_idx, 0:, head_idx, :], TensorLayout((br, head_size), (num_heads * head_size, 1)), "global"
                )
                txgq = partition_src(tg_q, auto_copy())
                tg_do = tensor_view(
                    do_1[batch_idx, 0:, head_idx, :],
                    TensorLayout((head_size, seqlen_q), (1, num_heads * head_size)),
                    "global",
                )
                txgdo = partition_src(tg_do, auto_copy())
                # tg_o = tensor_view(o[batch_size, 0:, head_idx, :], TensorLayout((head_size, seqlen_q), (1, num_heads * head_size)), "global")
                # txgo = partition_src(tg_o, auto_copy())

                copy(auto_copy((bc, head_size)), txgk, txsk)
                copy(auto_copy((bc, head_size)), txgv, txsv)
                copy(auto_copy((br, head_size)), txgq, txsq)

                cp_async_commit_group()

                tr_q = make_tensor("float16", layout_auto((br, inst_h * 2)), "register")
                tr_k = make_tensor("float16", layout_auto((bc, inst_h * 2)), "register")
                tr_v = make_tensor("float16", layout_auto((bc, inst_h * 2)), "register")
                txrq = partition_dst(tr_q, auto_copy())
                txrk = partition_dst(tr_k, auto_copy())
                txrv = partition_dst(tr_v, auto_copy())

                cp_async_wait_group(0)
                syncthreads()

                copy(auto_copy(), txSq[:, :, 0], txrq[:, :, 0])
                copy(auto_copy(), txSk[:, :, 0], txrk[:, :, 0])
                copy(auto_copy(), txSv[:, :, 0], txrv[:, :, 0])

                for i in grid(tr):
                    tr_qk = make_tensor("float32", auto_layout, "register")
                    fill(tr_qk, 0.0)

                    if i >= 1:
                        copy(auto_copy(), txSq[:, :, 0], txrq[:, :, 0])

                    copy(auto_copy((head_size, br)), txgdo[:, :, i], txsdo)
                    cp_async_commit_group()

                    # compute k*q^T
                    for hi in grid(h_tile_max, "u+"):
                        h_tile_next = (hi + 1) % h_tile_max
                        if hi < h_tile_max - 1:
                            copy(auto_copy(), txSq[:, :, hi + 1], txrq[:, :, (hi + 1) % 2])
                        copy(auto_copy(), txSk[:, :, h_tile_next], txrk[:, :, (hi + 1) % 2])
                        mma(tiled_mma_qk, tr_qk, txrk[:, :, hi % 2], txrq[:, :, hi % 2], tr_qk)

                    li = make_tensor("float32", layout_auto((bc, br), (0, 1)), "register")
                    mi = make_tensor("float32", layout_auto((bc, br), (0, 1)), "register")
                    tg_l = tensor_view(l[batch_idx, head_idx, i * br :], TensorLayout((bc, br), (0, 1)), "global")
                    tg_m = tensor_view(m[batch_idx, head_idx, i * br :], TensorLayout((bc, br), (0, 1)), "global")
                    txgli = partition_src(tg_l, auto_copy())
                    txgmi = partition_src(tg_m, auto_copy())
                    txrli = partition_dst(li, auto_copy())
                    txrmi = partition_dst(mi, auto_copy())
                    copy(auto_copy((bc, br)), txgli, txrli)
                    copy(auto_copy((bc, br)), txgmi, txrmi)

                    p = li * exp(tr_qk - mi)
                    p_f16 = cast(p, f16)
                    txrp = partition_A(p_f16, tiled_mma_pdo)

                    cp_async_wait_group(0)
                    syncthreads()

                    # compute p*do
                    tr_do = make_tensor("float16", layout_auto((head_size, inst_r * 2)), "register")
                    txrdo = partition_dst(tr_do, auto_copy())

                    copy(auto_copy(), txSdo[:, :, 0], txrdo[:, :, 0])
                    for ri in grid(r_tile_max, "u+"):
                        if ri < r_tile_max - 1:
                            copy(auto_copy(), txSdo[:, :, ri + 1], txrdo[:, :, (ri + 1) % 2])
                        mma(tiled_mma_pdo, tr_dv, txrp[:, :, ri], txrdo[:, :, ri % 2], tr_dv)

                    # compute v*do^T
                    tr_dp = make_tensor("float32", auto_layout, "register")
                    fill(tr_dp, 0.0)

                    tr_dot = make_tensor("float16", layout_auto((br, inst_h * 2)), "register")
                    txrdot = partition_dst(tr_dot, auto_copy())
                    copy(auto_copy(), txSdot[:, :, 0], txrdot[:, :, 0])

                    for hi in grid(h_tile_max, "u+"):
                        h_tile_next = (hi + 1) % h_tile_max
                        if hi < h_tile_max - 1:
                            copy(auto_copy(), txSdot[:, :, h_tile_next], txrdot[:, :, (hi + 1) % 2])
                        copy(auto_copy(), txSv[:, :, h_tile_next], txrv[:, :, (hi + 1) % 2])
                        mma(tiled_mma_qk, tr_dp, txrv[:, :, hi % 2], txrdot[:, :, hi % 2], tr_dp)

                    # tr_do1 = make_tensor("float16", auto_layout, "register")
                    # tr_o = make_tensor("float16", auto_layout, "register")
                    # txro = partition_dst(tr_o, auto_copy())
                    # copy(auto_copy((head_size, br)), txgo[:, :, i], txro)
                    # txsdo1 = partition_src(ts_do, auto_copy())
                    # txrdo1 = partition_dst(tr_do1, auto_copy())
                    # copy(auto_copy((head_size, br)), txsdo1, txrdo1)
                    # d = reduce_sum(tr_o * tr_do1, 0)

                    # di = broadcast_to(d, p_f16)
                    ds = p_f16 * (cast(tr_dp, f16))
                    txrds = partition_src(ds, auto_copy())
                    copy(auto_copy((bc, br)), txrds, txsds)
                    syncthreads()

                    # compute ds*Q^T
                    tr_ds1 = make_tensor("float16", layout_auto((bc, inst_r * 2)), "register")
                    tr_qt = make_tensor("float16", layout_auto((head_size, inst_r * 2)), "register")
                    txrds1 = partition_dst(tr_ds1, auto_copy())
                    txrqt = partition_dst(tr_qt, auto_copy())
                    copy(auto_copy(), txSds[:, :, 0], txrds1[:, :, 0])
                    copy(auto_copy(), txSqt[:, :, 0], txrqt[:, :, 0])
                    for ri in grid(r_tile_max, "u+"):
                        if ri < r_tile_max - 1:
                            copy(auto_copy(), txSds[:, :, ri + 1], txrds1[:, :, (ri + 1) % 2])
                            copy(auto_copy(), txSqt[:, :, ri + 1], txrqt[:, :, (ri + 1) % 2])
                        mma(tiled_mma_dsq, tr_dk, txrds1[:, :, ri % 2], txrqt[:, :, ri % 2], tr_dk)

                    syncthreads()
                    # copy q
                    if i < tr - 1:
                        tg_q1 = tensor_view(
                            q[batch_idx, (i + 1) * br :, head_idx, :],
                            TensorLayout((br, head_size), (num_heads * head_size, 1)),
                            "global",
                        )
                        txgq1 = partition_src(tg_q1, auto_copy())
                        copy(auto_copy((br, head_size)), txgq1, txsq[:, :])
                        cp_async_commit_group()

                    # compute ds^T*k^T
                    tr_dst = make_tensor("float16", layout_auto((br, inst_c * 2)), "register")
                    tr_kt = make_tensor("float16", layout_auto((head_size, inst_c * 2)), "register")
                    tr_dq = make_tensor("float32", auto_layout, "register")
                    fill(tr_dq, 0.0)
                    txrdst = partition_dst(tr_dst, auto_copy())
                    txrkt = partition_dst(tr_kt, auto_copy())
                    copy(auto_copy(), txSdst[:, :, 0], txrdst[:, :, 0])
                    copy(auto_copy(), txSkt[:, :, 0], txrkt[:, :, 0])

                    for ci in grid(c_tile_max, "u+"):
                        if ci < c_tile_max - 1:
                            copy(auto_copy(), txSdst[:, :, ci + 1], txrdst[:, :, (ci + 1) % 2])
                            copy(auto_copy(), txSkt[:, :, ci + 1], txrkt[:, :, (ci + 1) % 2])
                        mma(tiled_mma_dsk, tr_dq, txrdst[:, :, ci % 2], txrkt[:, :, ci % 2], tr_dq)

                    tr_dq_f16 = cast(tr_dq, f16)
                    # dq = dq + tr_dq
                    if j == 0:
                        tg_dq = tensor_view(
                            dq_seqk_parallel_parts[batch_idx, i * br :, head_idx, seqk_part, :],
                            TensorLayout((br, head_size), (num_heads * head_size, 1)),
                            "global",
                        )
                        txrdq = partition_src(tr_dq_f16, auto_copy())
                        txgdq = partition_src(tg_dq, auto_copy())
                        copy(auto_copy((br, head_size)), txrdq, txgdq)
                    else:
                        tg_dq = tensor_view(
                            dq_seqk_parallel_parts[batch_idx, i * br :, head_idx, seqk_part, :],
                            TensorLayout((br, head_size), (num_heads * head_size, 1)),
                            "global",
                        )
                        txgdq = partition_src(tg_dq, auto_copy())
                        tr_dq1 = make_tensor("float16", layout_auto((br, head_size)), "register")
                        txrdq1 = partition_dst(tr_dq1, auto_copy())
                        copy(auto_copy((br, head_size)), txgdq, txrdq1)
                        tr_dq1 = tr_dq1 + tr_dq_f16
                        txgdq2 = partition_dst(tg_dq, auto_copy())
                        copy(auto_copy((br, head_size)), txrdq1, txgdq2)

                    cp_async_wait_group(0)
                    syncthreads()

                # write dk, dv
                tg_dK = tensor_view(
                    dk[batch_idx, seqk_part * seqk_partition + j * bc :, head_idx, :],
                    TensorLayout((bc, head_size), (num_heads_k * head_size, 1)),
                    "global",
                )
                tg_dV = tensor_view(
                    dv[batch_idx, seqk_part * seqk_partition + j * bc :, head_idx, :],
                    TensorLayout((bc, head_size), (num_heads_k * head_size, 1)),
                    "global",
                )
                txgdk = partition_dst(tg_dK, auto_copy())
                txgdv = partition_dst(tg_dV, auto_copy())

                tr_dk_f16 = cast(tr_dk, f16)
                tr_dv_f16 = cast(tr_dv, f16)
                tr_dK = rearrange(tr_dk_f16, auto_layout, "register")
                tr_dV = rearrange(tr_dv_f16, auto_layout, "register")
                txrdk = partition_src(tr_dK, auto_copy())
                txrdv = partition_src(tr_dV, auto_copy())
                copy(auto_copy((bc, head_size)), txrdk, txgdk)
                copy(auto_copy((bc, head_size)), txrdv, txgdv)

    func = script_module.build()
    # a_mem = hidet.empty([m, k], device="cuda")
    # b_mem = hidet.empty([k, n], device="cuda")
    # c_mem = hidet.empty([m, n], device="cuda")
    # func(a_mem, b_mem, c_mem)
    return func


def flash_attention_v3_bwd(batch_size, seqlen_q, num_heads, head_size, seqlen_k, num_heads_k, num_parallel_seqk_parts):
    from hidet.lang.types import u32, i32, f16, f32
    from hidet.lang import attrs
    from hidet.lang import shared_tensor, register_tensor
    from hidet.lang.cuda import syncthreads, cp_async_wait_all
    from hidet.lang.cuda import blockIdx, threadIdx, dynamic_shared_memory
    from hidet.lang.cuda import cp_async, cp_async_commit_group, cp_async_wait_group

    tiled_mma_qk, tiled_mma_pdo, tiled_mma_dsq, tiled_mma_dsk = get_tiled_mma(head_size)

    k_shape, k_tv_layout = tiled_mma_qk.a_tv_layout()
    q_shape, q_tv_layout = tiled_mma_qk.b_tv_layout()
    qk_shape, qk_tv_layout = tiled_mma_qk.c_tv_layout()
    print(tiled_mma_qk.str_indented())

    qk_t, qk_v = canonicalize(qk_tv_layout)

    threads = qk_t.size()
    br, inst_h = q_shape
    bc, inst_h_ = k_shape
    assert inst_h == inst_h_

    print(tiled_mma_pdo.str_indented())
    p_shape, p_tv_layout = tiled_mma_pdo.a_tv_layout()
    do_shape, do_tv_layout = tiled_mma_pdo.b_tv_layout()
    dv_shape, dv_tv_layout = tiled_mma_pdo.c_tv_layout()
    print(p_shape, p_tv_layout)
    print(do_shape, do_tv_layout)
    print(dv_shape, dv_tv_layout)

    _, inst_r = p_shape

    print(tiled_mma_dsk.str_indented())
    dst_shape, dst_tv_layout = tiled_mma_dsk.a_tv_layout()
    kt_shape, kt_tv_layout = tiled_mma_dsk.b_tv_layout()
    dq_shape, dq_tv_layout = tiled_mma_dsk.c_tv_layout()
    print(dst_shape, dst_tv_layout)
    print(kt_shape, kt_tv_layout)
    print(dq_shape, dq_tv_layout)

    _, inst_c = dst_shape

    print(tiled_mma_dsq.str_indented())
    ds_shape, ds_tv_layout = tiled_mma_dsq.a_tv_layout()
    qt_shape, qt_tv_layout = tiled_mma_dsq.b_tv_layout()
    dk_shape, dk_tv_layout = tiled_mma_dsq.c_tv_layout()
    print(ds_shape, ds_tv_layout)
    print(qt_shape, qt_tv_layout)
    print(dk_shape, dk_tv_layout)

    from hidet.ir.cute.algorithm import auto_copy, auto_mma
    from hidet.ir.cute import auto_layout, layout_auto

    from hidet.utils.py import cdiv
    from hidet.lang import grid

    n_size = seqlen_q * num_heads
    bs = batch_size * num_heads
    assert num_heads == num_heads_k
    #    assert head_size == bn
    import sys

    float_max = f32.max_value
    seqk_partition = bc
    while seqk_partition * num_parallel_seqk_parts < seqlen_k:
        seqk_partition += seqk_partition

    tr = cdiv(seqlen_q, br)
    tc = cdiv(seqk_partition, bc)
    print(bc, br)

    with hidet.script_module() as script_module:

        @hidet.script
        def func(
            q: f16[batch_size, seqlen_q, num_heads, head_size],
            k: f16[batch_size, seqlen_k, num_heads_k, head_size],
            v: f16[batch_size, seqlen_k, num_heads_k, head_size],
            o: f16[batch_size, seqlen_q, num_heads, head_size],
            do_1: f16[batch_size, seqlen_q, num_heads, head_size],
            dq: f16[batch_size, seqlen_q, num_heads, head_size],
            dk: f16[batch_size, seqlen_k, num_heads_k, head_size],
            dv: f16[batch_size, seqlen_k, num_heads_k, head_size],
            l: f32[batch_size, num_heads, seqlen_q],
            m: f32[batch_size, num_heads, seqlen_q],
            dq_seqk_parallel_parts: f16[batch_size, num_parallel_seqk_parts, seqlen_q, num_heads, head_size],
            lock: i32[batch_size, num_heads],
        ):
            attrs.func_kind = "cuda_kernel"
            attrs.cuda.block_dim = threads
            attrs.cuda.grid_dim = num_parallel_seqk_parts, bs
            attrs.cuda.dynamic_smem_bytes = 0

            seqk_part = blockIdx.x
            pid = blockIdx.y
            batch_idx = pid // num_heads
            head_idx = pid % num_heads

            if seqk_part == 0 and threadIdx.x == 0:
                lock[batch_idx, head_idx] = 0

            # Q -> br x head_size
            # K -> bc x head_size
            # S -> bc x br
            # P -> bc x br
            # dV -> bc x head_size
            # dO -> head_size x br
            # O -> head_size x br
            # dP -> bc x br
            # dOt -> br x head_size
            # V -> bc x head_size
            # dS -> bc x br
            # Qt -> head_size x br
            # dSt -> br x bc
            # Kt -> head_size x bc

            h_tile_max = (head_size + inst_h - 1) // inst_h
            r_tile_max = (br + inst_r - 1) // inst_r
            c_tile_max = (bc + inst_c - 1) // inst_c
            tr_dv = make_tensor("float32", auto_layout, "register")
            tr_dk = make_tensor("float32", auto_layout, "register")

            ts_q = make_tensor("float16", TensorLayout((br, head_size), (head_size, 1)), "shared")
            txSq = partition_src(ts_q, auto_copy())
            txsq = partition_dst(ts_q, auto_copy())

            ts_k = make_tensor("float16", TensorLayout((bc, head_size), (head_size, 1)), "shared")
            txSk = partition_src(ts_k, auto_copy())
            txsk = partition_dst(ts_k, auto_copy())

            ts_do = make_tensor("float16", TensorLayout((head_size, br), (1, head_size)), "shared")
            txSdo = partition_src(ts_do, auto_copy())
            txsdo = partition_dst(ts_do, auto_copy())

            ts_dot = transpose(ts_do, 1, 0)
            txSdot = partition_src(ts_dot, auto_copy())

            ts_ds = make_tensor("float16", TensorLayout((bc, br), (br, 1)), "shared")
            txSds = partition_src(ts_ds, auto_copy())
            txsds = partition_dst(ts_ds, auto_copy())

            ts_p = make_tensor("float16", TensorLayout((bc, br), (br, 1)), "shared")
            txSp = partition_src(ts_p, auto_copy())
            txsp = partition_dst(ts_p, auto_copy())

            ts_dst = transpose(ts_ds, 1, 0)
            txSdst = partition_src(ts_dst, auto_copy())

            ts_qt = transpose(ts_q, 1, 0)
            txSqt = partition_src(ts_qt, auto_copy())

            ts_kt = transpose(ts_k, 1, 0)
            txSkt = partition_src(ts_kt, auto_copy())

            for j in grid(tc):
                fill(tr_dv, 0.0)
                fill(tr_dk, 0.0)

                tg_k = tensor_view(
                    k[batch_idx, seqk_part * seqk_partition + j * bc :, head_idx, :],
                    TensorLayout((bc, head_size), (num_heads_k * head_size, 1)),
                    "global",
                )
                txgk = partition_src(tg_k, auto_copy())
                tg_v = tensor_view(
                    v[batch_idx, seqk_part * seqk_partition + j * bc :, head_idx, :],
                    TensorLayout((bc, head_size), (num_heads_k * head_size, 1)),
                    "global",
                )
                txgv = partition_src(tg_v, auto_copy())

                tg_q = tensor_view(
                    q[batch_idx, 0:, head_idx, :], TensorLayout((br, head_size), (num_heads * head_size, 1)), "global"
                )
                txgq = partition_src(tg_q, auto_copy())
                tg_do = tensor_view(
                    do_1[batch_idx, 0:, head_idx, :],
                    TensorLayout((head_size, seqlen_q), (1, num_heads * head_size)),
                    "global",
                )
                txgdo = partition_src(tg_do, auto_copy())

                copy(auto_copy((bc, head_size)), txgk, txsk)
                copy(auto_copy((br, head_size)), txgq, txsq)
                copy(auto_copy((head_size, br)), txgdo[:, :, 0], txsdo)
                cp_async_commit_group()

                tr_q = make_tensor("float16", layout_auto((br, inst_h * 2)), "register")
                tr_k = make_tensor("float16", layout_auto((bc, inst_h * 2)), "register")
                tr_v = make_tensor("float16", layout_auto((bc, head_size)), "register")
                txrq = partition_dst(tr_q, auto_copy())
                txrk = partition_dst(tr_k, auto_copy())
                txrv = partition_dst(tr_v, auto_copy())

                for hi in range(h_tile_max):
                    copy(auto_copy(), txgv[:, :, hi], txrv[:, :, hi])

                cp_async_wait_group(0)
                syncthreads()

                copy(auto_copy(), txSq[:, :, 0], txrq[:, :, 0])
                copy(auto_copy(), txSk[:, :, 0], txrk[:, :, 0])

                for i in grid(tr):
                    tr_qk = make_tensor("float32", auto_layout, "register")
                    fill(tr_qk, 0.0)

                    if i >= 1:
                        copy(auto_copy(), txSq[:, :, 0], txrq[:, :, 0])

                    # compute k*q^T
                    for hi in grid(h_tile_max, "u+"):
                        h_tile_next = (hi + 1) % h_tile_max
                        if hi < h_tile_max - 1:
                            copy(auto_copy(), txSq[:, :, hi + 1], txrq[:, :, (hi + 1) % 2])
                        copy(auto_copy(), txSk[:, :, h_tile_next], txrk[:, :, (hi + 1) % 2])
                        mma(tiled_mma_qk, tr_qk, txrk[:, :, hi % 2], txrq[:, :, hi % 2], tr_qk)

                    li = make_tensor("float32", layout_auto((bc, br), (0, 1)), "register")
                    mi = make_tensor("float32", layout_auto((bc, br), (0, 1)), "register")
                    tg_l = tensor_view(l[batch_idx, head_idx, i * br :], TensorLayout((bc, br), (0, 1)), "global")
                    tg_m = tensor_view(m[batch_idx, head_idx, i * br :], TensorLayout((bc, br), (0, 1)), "global")
                    txgli = partition_src(tg_l, auto_copy())
                    txgmi = partition_src(tg_m, auto_copy())
                    txrli = partition_dst(li, auto_copy())
                    txrmi = partition_dst(mi, auto_copy())
                    copy(auto_copy((bc, br)), txgli, txrli)
                    copy(auto_copy((bc, br)), txgmi, txrmi)

                    p = exp(tr_qk - li)

                    # compute v*do^T
                    tr_dp = make_tensor("float32", auto_layout, "register")
                    fill(tr_dp, 0.0)

                    tr_dot = make_tensor("float16", layout_auto((br, inst_h * 2)), "register")
                    txrdot = partition_dst(tr_dot, auto_copy())
                    copy(auto_copy(), txSdot[:, :, 0], txrdot[:, :, 0])

                    for hi in grid(h_tile_max, "u+"):
                        h_tile_next = (hi + 1) % h_tile_max
                        if hi < h_tile_max - 1:
                            copy(auto_copy(), txSdot[:, :, h_tile_next], txrdot[:, :, (hi + 1) % 2])
                        # copy(auto_copy(), txSv[:, :, h_tile_next], txrv[:, :, (hi + 1) % 2])
                        mma(tiled_mma_qk, tr_dp, txrv[:, :, hi], txrdot[:, :, hi % 2], tr_dp)

                    ds = cast(p * tr_dp, f16)
                    txrds = partition_src(ds, auto_copy())
                    copy(auto_copy((bc, br)), txrds, txsds)

                    p_f16 = cast(p, f16)
                    txrp = partition_src(p_f16, auto_copy())
                    copy(auto_copy((bc, br)), txrp, txsp)

                    syncthreads()

                    # compute p*do
                    tr_p = make_tensor("float16", layout_auto((bc, inst_r * 2)), "register")
                    txrp1 = partition_dst(tr_p, auto_copy())
                    tr_do = make_tensor("float16", layout_auto((head_size, inst_r * 2)), "register")
                    txrdo = partition_dst(tr_do, auto_copy())

                    copy(auto_copy(), txSp[:, :, 0], txrp1[:, :, 0])
                    copy(auto_copy(), txSdo[:, :, 0], txrdo[:, :, 0])
                    for ri in grid(r_tile_max, "u+"):
                        if ri < r_tile_max - 1:
                            copy(auto_copy(), txSp[:, :, ri + 1], txrp1[:, :, (ri + 1) % 2])
                            copy(auto_copy(), txSdo[:, :, ri + 1], txrdo[:, :, (ri + 1) % 2])
                        mma(tiled_mma_pdo, tr_dv, txrp1[:, :, ri % 2], txrdo[:, :, ri % 2], tr_dv)

                    syncthreads()

                    if i < tr - 1:
                        copy(auto_copy((head_size, br)), txgdo[:, :, i + 1], txsdo)
                        cp_async_commit_group()

                    # compute ds*Q^T
                    tr_ds1 = make_tensor("float16", layout_auto((bc, inst_r * 2)), "register")
                    tr_qt = make_tensor("float16", layout_auto((head_size, inst_r * 2)), "register")
                    txrds1 = partition_dst(tr_ds1, auto_copy())
                    txrqt = partition_dst(tr_qt, auto_copy())
                    copy(auto_copy(), txSds[:, :, 0], txrds1[:, :, 0])
                    copy(auto_copy(), txSqt[:, :, 0], txrqt[:, :, 0])
                    for ri in grid(r_tile_max, "u+"):
                        if ri < r_tile_max - 1:
                            copy(auto_copy(), txSds[:, :, ri + 1], txrds1[:, :, (ri + 1) % 2])
                            copy(auto_copy(), txSqt[:, :, ri + 1], txrqt[:, :, (ri + 1) % 2])
                        mma(tiled_mma_dsq, tr_dk, txrds1[:, :, ri % 2], txrqt[:, :, ri % 2], tr_dk)

                    # compute ds^T*k^T
                    tr_dst = make_tensor("float16", layout_auto((br, inst_c * 2)), "register")
                    tr_kt = make_tensor("float16", layout_auto((head_size, inst_c * 2)), "register")
                    tr_dq = make_tensor("float32", auto_layout, "register")
                    fill(tr_dq, 0.0)
                    txrdst = partition_dst(tr_dst, auto_copy())
                    txrkt = partition_dst(tr_kt, auto_copy())
                    copy(auto_copy(), txSdst[:, :, 0], txrdst[:, :, 0])
                    copy(auto_copy(), txSkt[:, :, 0], txrkt[:, :, 0])

                    syncthreads()
                    # copy q
                    if i < tr - 1:
                        tg_q1 = tensor_view(
                            q[batch_idx, (i + 1) * br :, head_idx, :],
                            TensorLayout((br, head_size), (num_heads * head_size, 1)),
                            "global",
                        )
                        txgq1 = partition_src(tg_q1, auto_copy())
                        copy(auto_copy((br, head_size)), txgq1, txsq[:, :])
                        cp_async_commit_group()

                    for ci in grid(c_tile_max, "u+"):
                        if ci < c_tile_max - 1:
                            copy(auto_copy(), txSdst[:, :, ci + 1], txrdst[:, :, (ci + 1) % 2])
                            copy(auto_copy(), txSkt[:, :, ci + 1], txrkt[:, :, (ci + 1) % 2])
                        mma(tiled_mma_dsk, tr_dq, txrdst[:, :, ci % 2], txrkt[:, :, ci % 2], tr_dq)

                    tr_dq_f16 = cast(tr_dq, f16)
                    tr_dq_contig = rearrange(tr_dq_f16, auto_layout, "register")
                    # dq = dq + tr_dq
                    if j == 0:
                        tg_dq = tensor_view(
                            dq_seqk_parallel_parts[batch_idx, seqk_part, i * br :, head_idx, :],
                            TensorLayout((br, head_size), (num_heads * head_size, 1)),
                            "global",
                        )
                        txrdq = partition_src(tr_dq_contig, auto_copy())
                        txgdq = partition_src(tg_dq, auto_copy())
                        copy(auto_copy((br, head_size)), txrdq, txgdq)
                    else:
                        tg_dq = tensor_view(
                            dq_seqk_parallel_parts[batch_idx, seqk_part, i * br :, head_idx, :],
                            TensorLayout((br, head_size), (num_heads * head_size, 1)),
                            "global",
                        )
                        txgdq = partition_src(tg_dq, auto_copy())
                        tr_dq1 = make_tensor("float16", layout_auto((br, head_size)), "register")
                        txrdq1 = partition_dst(tr_dq1, auto_copy())
                        copy(auto_copy((br, head_size)), txgdq, txrdq1)
                        tr_dq_contig = tr_dq_contig + tr_dq1
                        txrdq = partition_src(tr_dq_contig, auto_copy())
                        txgdq2 = partition_dst(tg_dq, auto_copy())
                        copy(auto_copy((br, head_size)), txrdq, txgdq2)
                    cp_async_wait_group(0)
                    syncthreads()

                # write dk, dv
                tg_dK = tensor_view(
                    dk[batch_idx, seqk_part * seqk_partition + j * bc :, head_idx, :],
                    TensorLayout((bc, head_size), (num_heads_k * head_size, 1)),
                    "global",
                )
                tg_dV = tensor_view(
                    dv[batch_idx, seqk_part * seqk_partition + j * bc :, head_idx, :],
                    TensorLayout((bc, head_size), (num_heads_k * head_size, 1)),
                    "global",
                )
                txgdk = partition_dst(tg_dK, auto_copy())
                txgdv = partition_dst(tg_dV, auto_copy())

                tr_dk_f16 = cast(tr_dk, f16)
                tr_dv_f16 = cast(tr_dv, f16)
                tr_dK = rearrange(tr_dk_f16, auto_layout, "register")
                tr_dV = rearrange(tr_dv_f16, auto_layout, "register")
                txrdk = partition_src(tr_dK, auto_copy())
                txrdv = partition_src(tr_dV, auto_copy())
                copy(auto_copy((bc, head_size)), txrdk, txgdk)
                copy(auto_copy((bc, head_size)), txrdv, txgdv)

            counter = ~lock[batch_idx, head_idx]
            if threadIdx.x == 0:
                atomic_add(counter, 1)
            if seqk_part == 0:
                acquire_seq_semaphore(counter, num_parallel_seqk_parts)
                for i in grid(tr):
                    tr_dq = make_tensor("float16", layout_auto((num_parallel_seqk_parts, br, head_size)), "register")
                    fill(tr_dq, 0.0)
                    tg_dq = tensor_view(
                        dq_seqk_parallel_parts[batch_idx, :, i * br :, head_idx, :],
                        TensorLayout(
                            (num_parallel_seqk_parts, br, head_size),
                            (seqlen_q * num_heads * head_size, num_heads * head_size, 1),
                        ),
                        "global",
                    )
                    txgdq = partition_src(tg_dq, auto_copy())
                    txrdq = partition_dst(tr_dq, auto_copy())
                    copy(auto_copy((num_parallel_seqk_parts, br, head_size)), txgdq, txrdq)
                    tr_dq_final = reduce_sum(tr_dq, 0)
                    tg_dq_final = tensor_view(
                        dq[batch_idx, i * br :, head_idx, :],
                        TensorLayout((num_parallel_seqk_parts, br, head_size), (0, num_heads * head_size, 1)),
                        "global",
                    )
                    txrdq1 = partition_src(tr_dq_final, auto_copy())
                    txgdq1 = partition_dst(tg_dq_final, auto_copy())
                    copy(auto_copy((num_parallel_seqk_parts, br, head_size)), txrdq1, txgdq1)

    func = script_module.build()
    # a_mem = hidet.empty([m, k], device="cuda")
    # b_mem = hidet.empty([k, n], device="cuda")
    # c_mem = hidet.empty([m, n], device="cuda")
    # func(a_mem, b_mem, c_mem)
    return func


# double-buffer Q
def flash_attention_v4_bwd(batch_size, seqlen_q, num_heads, head_size, seqlen_k, num_heads_k, num_parallel_seqk_parts):
    from hidet.lang.types import u32, i32, f16, f32
    from hidet.lang import attrs
    from hidet.lang import shared_tensor, register_tensor
    from hidet.lang.cuda import syncthreads, cp_async_wait_all
    from hidet.lang.cuda import blockIdx, threadIdx, dynamic_shared_memory
    from hidet.lang.cuda import cp_async, cp_async_commit_group, cp_async_wait_group

    tiled_mma_qk, tiled_mma_pdo, tiled_mma_dsq, tiled_mma_dsk = get_tiled_mma(head_size)

    k_shape, k_tv_layout = tiled_mma_qk.a_tv_layout()
    q_shape, q_tv_layout = tiled_mma_qk.b_tv_layout()
    qk_shape, qk_tv_layout = tiled_mma_qk.c_tv_layout()
    print(tiled_mma_qk.str_indented())

    qk_t, qk_v = canonicalize(qk_tv_layout)

    threads = qk_t.size()
    br, inst_h = q_shape
    bc, inst_h_ = k_shape
    assert inst_h == inst_h_

    print(tiled_mma_pdo.str_indented())
    p_shape, p_tv_layout = tiled_mma_pdo.a_tv_layout()
    do_shape, do_tv_layout = tiled_mma_pdo.b_tv_layout()
    dv_shape, dv_tv_layout = tiled_mma_pdo.c_tv_layout()
    print(p_shape, p_tv_layout)
    print(do_shape, do_tv_layout)
    print(dv_shape, dv_tv_layout)

    _, inst_r = p_shape

    print(tiled_mma_dsk.str_indented())
    dst_shape, dst_tv_layout = tiled_mma_dsk.a_tv_layout()
    kt_shape, kt_tv_layout = tiled_mma_dsk.b_tv_layout()
    dq_shape, dq_tv_layout = tiled_mma_dsk.c_tv_layout()
    print(dst_shape, dst_tv_layout)
    print(kt_shape, kt_tv_layout)
    print(dq_shape, dq_tv_layout)

    _, inst_c = dst_shape

    print(tiled_mma_dsq.str_indented())
    ds_shape, ds_tv_layout = tiled_mma_dsq.a_tv_layout()
    qt_shape, qt_tv_layout = tiled_mma_dsq.b_tv_layout()
    dk_shape, dk_tv_layout = tiled_mma_dsq.c_tv_layout()
    print(ds_shape, ds_tv_layout)
    print(qt_shape, qt_tv_layout)
    print(dk_shape, dk_tv_layout)

    from hidet.ir.cute.algorithm import auto_copy, auto_mma
    from hidet.ir.cute import auto_layout, layout_auto

    from hidet.utils.py import cdiv
    from hidet.lang import grid

    n_size = seqlen_q * num_heads
    bs = batch_size * num_heads
    assert num_heads == num_heads_k
    #    assert head_size == bn
    import sys

    float_max = f32.max_value
    seqk_partition = bc
    while seqk_partition * num_parallel_seqk_parts < seqlen_k:
        seqk_partition += seqk_partition

    tr = cdiv(seqlen_q, br)
    tc = cdiv(seqk_partition, bc)
    print(bc, br)

    with hidet.script_module() as script_module:

        @hidet.script
        def func(
            q: f16[batch_size, seqlen_q, num_heads, head_size],
            k: f16[batch_size, seqlen_k, num_heads_k, head_size],
            v: f16[batch_size, seqlen_k, num_heads_k, head_size],
            o: f16[batch_size, seqlen_q, num_heads, head_size],
            do_1: f16[batch_size, seqlen_q, num_heads, head_size],
            dq: f16[batch_size, seqlen_q, num_heads, head_size],
            dk: f16[batch_size, seqlen_k, num_heads_k, head_size],
            dv: f16[batch_size, seqlen_k, num_heads_k, head_size],
            l: f32[batch_size, num_heads, seqlen_q],
            m: f32[batch_size, num_heads, seqlen_q],
            dq_seqk_parallel_parts: f16[batch_size, num_parallel_seqk_parts, seqlen_q, num_heads, head_size],
            lock: i32[batch_size, num_heads],
            dp_sum: f32[batch_size, num_heads, seqlen_q],
        ):
            attrs.func_kind = "cuda_kernel"
            attrs.cuda.block_dim = threads
            attrs.cuda.grid_dim = num_parallel_seqk_parts, bs
            attrs.cuda.dynamic_smem_bytes = 0

            seqk_part = blockIdx.x
            pid = blockIdx.y
            batch_idx = pid // num_heads
            head_idx = pid % num_heads

            if seqk_part == 0 and threadIdx.x == 0:
                lock[batch_idx, head_idx] = 0

            # Q -> br x head_size
            # K -> bc x head_size
            # S -> bc x br
            # P -> bc x br
            # dV -> bc x head_size
            # dO -> head_size x br
            # O -> head_size x br
            # dP -> bc x br
            # dOt -> br x head_size
            # V -> bc x head_size
            # dS -> bc x br
            # Qt -> head_size x br
            # dSt -> br x bc
            # Kt -> head_size x bc

            h_tile_max = (head_size + inst_h - 1) // inst_h
            r_tile_max = (br + inst_r - 1) // inst_r
            c_tile_max = (bc + inst_c - 1) // inst_c
            tr_dv = make_tensor("float32", auto_layout, "register")
            tr_dk = make_tensor("float32", auto_layout, "register")

            ts_q = make_tensor("float16", TensorLayout((br, head_size, 2), (head_size, 1, br * head_size)), "shared")
            txSq = partition_src(ts_q, auto_copy())
            txsq = partition_dst(ts_q, auto_copy())

            ts_k = make_tensor("float16", TensorLayout((bc, head_size), (head_size, 1)), "shared")
            txSk = partition_src(ts_k, auto_copy())
            txsk = partition_dst(ts_k, auto_copy())

            ts_do = make_tensor("float16", TensorLayout((head_size, br), (1, head_size)), "shared")
            txSdo = partition_src(ts_do, auto_copy())
            txsdo = partition_dst(ts_do, auto_copy())

            ts_dot = transpose(ts_do, 1, 0)
            txSdot = partition_src(ts_dot, auto_copy())

            ts_ds = make_tensor("float16", TensorLayout((bc, br), (br, 1)), "shared")
            txSds = partition_src(ts_ds, auto_copy())
            txsds = partition_dst(ts_ds, auto_copy())

            ts_p = make_tensor("float16", TensorLayout((bc, br), (br, 1)), "shared")
            txSp = partition_src(ts_p, auto_copy())
            txsp = partition_dst(ts_p, auto_copy())

            ts_dst = transpose(ts_ds, 1, 0)
            txSdst = partition_src(ts_dst, auto_copy())

            ts_qt = transpose(ts_q, 1, 0, 2)
            txSqt = partition_src(ts_qt, auto_copy())

            ts_kt = transpose(ts_k, 1, 0)
            txSkt = partition_src(ts_kt, auto_copy())

            for j in grid(tc):
                fill(tr_dv, 0.0)
                fill(tr_dk, 0.0)

                tg_k = tensor_view(
                    k[batch_idx, seqk_part * seqk_partition + j * bc :, head_idx, :],
                    TensorLayout((bc, head_size), (num_heads_k * head_size, 1)),
                    "global",
                )
                txgk = partition_src(tg_k, auto_copy())
                tg_v = tensor_view(
                    v[batch_idx, seqk_part * seqk_partition + j * bc :, head_idx, :],
                    TensorLayout((bc, head_size), (num_heads_k * head_size, 1)),
                    "global",
                )
                txgv = partition_src(tg_v, auto_copy())

                tg_q = tensor_view(
                    q[batch_idx, 0:, head_idx, :], TensorLayout((br, head_size), (num_heads * head_size, 1)), "global"
                )
                txgq = partition_src(tg_q, auto_copy())
                tg_do = tensor_view(
                    do_1[batch_idx, 0:, head_idx, :],
                    TensorLayout((head_size, seqlen_q), (1, num_heads * head_size)),
                    "global",
                )
                txgdo = partition_src(tg_do, auto_copy())

                tg_dp_sum = tensor_view(dp_sum[batch_idx, head_idx, 0:], TensorLayout((bc, seqlen_q), (0, 1)), "global")
                txgdpsum = partition_src(tg_dp_sum, auto_copy())

                copy(auto_copy((bc, head_size)), txgk, txsk)
                copy(auto_copy((br, head_size)), txgq, txsq[:, :, 0])
                copy(auto_copy((head_size, br)), txgdo[:, :, 0], txsdo)
                cp_async_commit_group()

                tr_q = make_tensor("float16", layout_auto((br, inst_h * 2)), "register")
                tr_k = make_tensor("float16", layout_auto((bc, inst_h * 2)), "register")
                tr_v = make_tensor("float16", layout_auto((bc, head_size)), "register")
                txrq = partition_dst(tr_q, auto_copy())
                txrk = partition_dst(tr_k, auto_copy())
                txrv = partition_dst(tr_v, auto_copy())

                for hi in range(h_tile_max):
                    copy(auto_copy(), txgv[:, :, hi], txrv[:, :, hi])

                cp_async_wait_group(0)
                syncthreads()

                copy(auto_copy(), txSq[:, :, 0, 0], txrq[:, :, 0])
                copy(auto_copy(), txSk[:, :, 0], txrk[:, :, 0])

                q_write_idx = 1
                q_read_idx = 0

                for i in grid(tr):
                    tr_qk = make_tensor("float32", auto_layout, "register")
                    fill(tr_qk, 0.0)
                    tr_dp_sum = make_tensor("float32", layout_auto((bc, br), (0, 1)), "register")
                    txrdpsum = partition_dst(tr_dp_sum, auto_copy())
                    copy(auto_copy((bc, br)), txgdpsum[:, :, i], txrdpsum)

                    if i >= 1:
                        copy(auto_copy(), txSq[:, :, 0, q_read_idx], txrq[:, :, 0])

                    # compute k*q^T
                    for hi in grid(h_tile_max, "u+"):
                        h_tile_next = (hi + 1) % h_tile_max
                        if hi < h_tile_max - 1:
                            copy(auto_copy(), txSq[:, :, hi + 1, q_read_idx], txrq[:, :, (hi + 1) % 2])
                        copy(auto_copy(), txSk[:, :, h_tile_next], txrk[:, :, (hi + 1) % 2])
                        mma(tiled_mma_qk, tr_qk, txrk[:, :, hi % 2], txrq[:, :, hi % 2], tr_qk)

                    li = make_tensor("float32", layout_auto((bc, br), (0, 1)), "register")
                    mi = make_tensor("float32", layout_auto((bc, br), (0, 1)), "register")
                    tg_l = tensor_view(l[batch_idx, head_idx, i * br :], TensorLayout((bc, br), (0, 1)), "global")
                    tg_m = tensor_view(m[batch_idx, head_idx, i * br :], TensorLayout((bc, br), (0, 1)), "global")
                    txgli = partition_src(tg_l, auto_copy())
                    txgmi = partition_src(tg_m, auto_copy())
                    txrli = partition_dst(li, auto_copy())
                    txrmi = partition_dst(mi, auto_copy())
                    copy(auto_copy((bc, br)), txgli, txrli)
                    copy(auto_copy((bc, br)), txgmi, txrmi)

                    p = exp(tr_qk - li)

                    # compute v*do^T
                    tr_dp = make_tensor("float32", auto_layout, "register")
                    fill(tr_dp, 0.0)

                    tr_dot = make_tensor("float16", layout_auto((br, inst_h * 2)), "register")
                    txrdot = partition_dst(tr_dot, auto_copy())
                    copy(auto_copy(), txSdot[:, :, 0], txrdot[:, :, 0])

                    for hi in grid(h_tile_max, "u+"):
                        h_tile_next = (hi + 1) % h_tile_max
                        if hi < h_tile_max - 1:
                            copy(auto_copy(), txSdot[:, :, h_tile_next], txrdot[:, :, (hi + 1) % 2])
                        # copy(auto_copy(), txSv[:, :, h_tile_next], txrv[:, :, (hi + 1) % 2])
                        mma(tiled_mma_qk, tr_dp, txrv[:, :, hi], txrdot[:, :, hi % 2], tr_dp)

                    ds = cast(p * (tr_dp - tr_dp_sum), f16)
                    txrds = partition_src(ds, auto_copy())
                    copy(auto_copy((bc, br)), txrds, txsds)

                    p_f16 = cast(p, f16)
                    txrp = partition_src(p_f16, auto_copy())
                    copy(auto_copy((bc, br)), txrp, txsp)

                    syncthreads()

                    # compute p*do
                    tr_p = make_tensor("float16", layout_auto((bc, inst_r * 2)), "register")
                    txrp1 = partition_dst(tr_p, auto_copy())
                    tr_do = make_tensor("float16", layout_auto((head_size, inst_r * 2)), "register")
                    txrdo = partition_dst(tr_do, auto_copy())

                    copy(auto_copy(), txSp[:, :, 0], txrp1[:, :, 0])
                    copy(auto_copy(), txSdo[:, :, 0], txrdo[:, :, 0])
                    for ri in grid(r_tile_max, "u+"):
                        if ri < r_tile_max - 1:
                            copy(auto_copy(), txSp[:, :, ri + 1], txrp1[:, :, (ri + 1) % 2])
                            copy(auto_copy(), txSdo[:, :, ri + 1], txrdo[:, :, (ri + 1) % 2])
                        mma(tiled_mma_pdo, tr_dv, txrp1[:, :, ri % 2], txrdo[:, :, ri % 2], tr_dv)

                    syncthreads()

                    if i < tr - 1:
                        copy(auto_copy((head_size, br)), txgdo[:, :, i + 1], txsdo)
                        tg_q1 = tensor_view(
                            q[batch_idx, (i + 1) * br :, head_idx, :],
                            TensorLayout((br, head_size), (num_heads * head_size, 1)),
                            "global",
                        )
                        txgq1 = partition_src(tg_q1, auto_copy())
                        copy(auto_copy((br, head_size)), txgq1, txsq[:, :, q_write_idx])
                        cp_async_commit_group()

                    # compute ds^T*k^T
                    tr_dst = make_tensor("float16", layout_auto((br, inst_c * 2)), "register")
                    tr_kt = make_tensor("float16", layout_auto((head_size, inst_c * 2)), "register")
                    tr_dq = make_tensor("float32", auto_layout, "register")
                    fill(tr_dq, 0.0)
                    txrdst = partition_dst(tr_dst, auto_copy())
                    txrkt = partition_dst(tr_kt, auto_copy())
                    copy(auto_copy(), txSdst[:, :, 0], txrdst[:, :, 0])
                    copy(auto_copy(), txSkt[:, :, 0], txrkt[:, :, 0])
                    for ci in grid(c_tile_max, "u+"):
                        if ci < c_tile_max - 1:
                            copy(auto_copy(), txSdst[:, :, ci + 1], txrdst[:, :, (ci + 1) % 2])
                            copy(auto_copy(), txSkt[:, :, ci + 1], txrkt[:, :, (ci + 1) % 2])
                        mma(tiled_mma_dsk, tr_dq, txrdst[:, :, ci % 2], txrkt[:, :, ci % 2], tr_dq)

                    tr_dq_f16 = cast(tr_dq, f16)
                    # dq = dq + tr_dq
                    tg_dq = tensor_view(
                        dq[batch_idx, i * br :, head_idx, :],
                        TensorLayout((br, head_size), (num_heads * head_size, 1)),
                        "global",
                    )
                    cute_atomic_add(tr_dq_f16, tg_dq)

                    # compute ds*Q^T
                    tr_ds1 = make_tensor("float16", layout_auto((bc, inst_r * 2)), "register")
                    tr_qt = make_tensor("float16", layout_auto((head_size, inst_r * 2)), "register")
                    txrds1 = partition_dst(tr_ds1, auto_copy())
                    txrqt = partition_dst(tr_qt, auto_copy())
                    copy(auto_copy(), txSds[:, :, 0], txrds1[:, :, 0])
                    copy(auto_copy(), txSqt[:, :, 0, q_read_idx], txrqt[:, :, 0])
                    for ri in grid(r_tile_max, "u+"):
                        if ri < r_tile_max - 1:
                            copy(auto_copy(), txSds[:, :, ri + 1], txrds1[:, :, (ri + 1) % 2])
                            copy(auto_copy(), txSqt[:, :, ri + 1, q_read_idx], txrqt[:, :, (ri + 1) % 2])
                        mma(tiled_mma_dsq, tr_dk, txrds1[:, :, ri % 2], txrqt[:, :, ri % 2], tr_dk)

                    cp_async_wait_group(0)
                    syncthreads()

                    q_read_idx = 1 - q_read_idx
                    q_write_idx = 1 - q_write_idx
                # write dk, dv
                tg_dK = tensor_view(
                    dk[batch_idx, seqk_part * seqk_partition + j * bc :, head_idx, :],
                    TensorLayout((bc, head_size), (num_heads_k * head_size, 1)),
                    "global",
                )
                tg_dV = tensor_view(
                    dv[batch_idx, seqk_part * seqk_partition + j * bc :, head_idx, :],
                    TensorLayout((bc, head_size), (num_heads_k * head_size, 1)),
                    "global",
                )
                txgdk = partition_dst(tg_dK, auto_copy())
                txgdv = partition_dst(tg_dV, auto_copy())

                tr_dk_f16 = cast(tr_dk, f16)
                tr_dv_f16 = cast(tr_dv, f16)
                tr_dK = rearrange(tr_dk_f16, auto_layout, "register")
                tr_dV = rearrange(tr_dv_f16, auto_layout, "register")
                txrdk = partition_src(tr_dK, auto_copy())
                txrdv = partition_src(tr_dV, auto_copy())
                copy(auto_copy((bc, head_size)), txrdk, txgdk)
                copy(auto_copy((bc, head_size)), txrdv, txgdv)

    func = script_module.build()
    # a_mem = hidet.empty([m, k], device="cuda")
    # b_mem = hidet.empty([k, n], device="cuda")
    # c_mem = hidet.empty([m, n], device="cuda")
    # func(a_mem, b_mem, c_mem)
    return func


# single-buffer Q to save shared memory consumption
def flash_attention_v5_bwd(batch_size, seqlen_q, num_heads, head_size, seqlen_k, num_heads_k, num_parallel_seqk_parts):
    from hidet.lang.types import u32, i32, f16, f32
    from hidet.lang import attrs
    from hidet.lang import shared_tensor, register_tensor
    from hidet.lang.cuda import syncthreads, cp_async_wait_all
    from hidet.lang.cuda import blockIdx, threadIdx, dynamic_shared_memory
    from hidet.lang.cuda import cp_async, cp_async_commit_group, cp_async_wait_group

    tiled_mma_qk, tiled_mma_pdo, tiled_mma_dsq, tiled_mma_dsk = get_tiled_mma(head_size)

    k_shape, k_tv_layout = tiled_mma_qk.a_tv_layout()
    q_shape, q_tv_layout = tiled_mma_qk.b_tv_layout()
    qk_shape, qk_tv_layout = tiled_mma_qk.c_tv_layout()
    print(tiled_mma_qk.str_indented())

    qk_t, qk_v = canonicalize(qk_tv_layout)

    threads = qk_t.size()
    br, inst_h = q_shape
    bc, inst_h_ = k_shape
    assert inst_h == inst_h_

    print(tiled_mma_pdo.str_indented())
    p_shape, p_tv_layout = tiled_mma_pdo.a_tv_layout()
    do_shape, do_tv_layout = tiled_mma_pdo.b_tv_layout()
    dv_shape, dv_tv_layout = tiled_mma_pdo.c_tv_layout()
    print(p_shape, p_tv_layout)
    print(do_shape, do_tv_layout)
    print(dv_shape, dv_tv_layout)

    _, inst_r = p_shape

    print(tiled_mma_dsk.str_indented())
    dst_shape, dst_tv_layout = tiled_mma_dsk.a_tv_layout()
    kt_shape, kt_tv_layout = tiled_mma_dsk.b_tv_layout()
    dq_shape, dq_tv_layout = tiled_mma_dsk.c_tv_layout()
    print(dst_shape, dst_tv_layout)
    print(kt_shape, kt_tv_layout)
    print(dq_shape, dq_tv_layout)

    _, inst_c = dst_shape

    print(tiled_mma_dsq.str_indented())
    ds_shape, ds_tv_layout = tiled_mma_dsq.a_tv_layout()
    qt_shape, qt_tv_layout = tiled_mma_dsq.b_tv_layout()
    dk_shape, dk_tv_layout = tiled_mma_dsq.c_tv_layout()
    print(ds_shape, ds_tv_layout)
    print(qt_shape, qt_tv_layout)
    print(dk_shape, dk_tv_layout)

    from hidet.ir.cute.algorithm import auto_copy, auto_mma
    from hidet.ir.cute import auto_layout, layout_auto

    from hidet.utils.py import cdiv
    from hidet.lang import grid

    n_size = seqlen_q * num_heads
    bs = batch_size * num_heads
    assert num_heads == num_heads_k
    #    assert head_size == bn
    import sys

    float_max = f32.max_value
    seqk_partition = bc
    while seqk_partition * num_parallel_seqk_parts < seqlen_k:
        seqk_partition += seqk_partition

    tr = cdiv(seqlen_q, br)
    tc = cdiv(seqk_partition, bc)
    print(bc, br)

    with hidet.script_module() as script_module:

        @hidet.script
        def func(
            q: f16[batch_size, seqlen_q, num_heads, head_size],
            k: f16[batch_size, seqlen_k, num_heads_k, head_size],
            v: f16[batch_size, seqlen_k, num_heads_k, head_size],
            o: f16[batch_size, seqlen_q, num_heads, head_size],
            do_1: f16[batch_size, seqlen_q, num_heads, head_size],
            dq: f16[batch_size, seqlen_q, num_heads, head_size],
            dk: f16[batch_size, seqlen_k, num_heads_k, head_size],
            dv: f16[batch_size, seqlen_k, num_heads_k, head_size],
            l: f32[batch_size, num_heads, seqlen_q],
            m: f32[batch_size, num_heads, seqlen_q],
            dq_seqk_parallel_parts: f16[batch_size, num_parallel_seqk_parts, seqlen_q, num_heads, head_size],
            lock: i32[batch_size, num_heads],
            dp_sum: f32[batch_size, num_heads, seqlen_q],
        ):
            attrs.func_kind = "cuda_kernel"
            attrs.cuda.block_dim = threads
            attrs.cuda.grid_dim = num_parallel_seqk_parts, bs
            attrs.cuda.dynamic_smem_bytes = 0

            seqk_part = blockIdx.x
            pid = blockIdx.y
            batch_idx = pid // num_heads
            head_idx = pid % num_heads

            if seqk_part == 0 and threadIdx.x == 0:
                lock[batch_idx, head_idx] = 0

            # Q -> br x head_size
            # K -> bc x head_size
            # S -> bc x br
            # P -> bc x br
            # dV -> bc x head_size
            # dO -> head_size x br
            # O -> head_size x br
            # dP -> bc x br
            # dOt -> br x head_size
            # V -> bc x head_size
            # dS -> bc x br
            # Qt -> head_size x br
            # dSt -> br x bc
            # Kt -> head_size x bc

            h_tile_max = (head_size + inst_h - 1) // inst_h
            r_tile_max = (br + inst_r - 1) // inst_r
            c_tile_max = (bc + inst_c - 1) // inst_c
            tr_dv = make_tensor("float32", auto_layout, "register")
            tr_dk = make_tensor("float32", auto_layout, "register")

            ts_q = make_tensor("float16", TensorLayout((br, head_size), (head_size, 1)), "shared")
            txSq = partition_src(ts_q, auto_copy())
            txsq = partition_dst(ts_q, auto_copy())

            ts_k = make_tensor("float16", TensorLayout((bc, head_size), (head_size, 1)), "shared")
            txSk = partition_src(ts_k, auto_copy())
            txsk = partition_dst(ts_k, auto_copy())

            ts_do = make_tensor("float16", TensorLayout((head_size, br), (1, head_size)), "shared")
            txSdo = partition_src(ts_do, auto_copy())
            txsdo = partition_dst(ts_do, auto_copy())

            ts_dot = transpose(ts_do, 1, 0)
            txSdot = partition_src(ts_dot, auto_copy())

            ts_ds = make_tensor("float16", TensorLayout((bc, br), (br, 1)), "shared")
            txSds = partition_src(ts_ds, auto_copy())
            txsds = partition_dst(ts_ds, auto_copy())

            ts_p = make_tensor("float16", TensorLayout((bc, br), (br, 1)), "shared")
            txSp = partition_src(ts_p, auto_copy())
            txsp = partition_dst(ts_p, auto_copy())

            ts_dst = transpose(ts_ds, 1, 0)
            txSdst = partition_src(ts_dst, auto_copy())

            ts_qt = transpose(ts_q, 1, 0)
            txSqt = partition_src(ts_qt, auto_copy())

            ts_kt = transpose(ts_k, 1, 0)
            txSkt = partition_src(ts_kt, auto_copy())

            for j in grid(tc):
                fill(tr_dv, 0.0)
                fill(tr_dk, 0.0)

                tg_k = tensor_view(
                    k[batch_idx, seqk_part * seqk_partition + j * bc :, head_idx, :],
                    TensorLayout((bc, head_size), (num_heads_k * head_size, 1)),
                    "global",
                )
                txgk = partition_src(tg_k, auto_copy())
                tg_v = tensor_view(
                    v[batch_idx, seqk_part * seqk_partition + j * bc :, head_idx, :],
                    TensorLayout((bc, head_size), (num_heads_k * head_size, 1)),
                    "global",
                )
                txgv = partition_src(tg_v, auto_copy())

                tg_q = tensor_view(
                    q[batch_idx, 0:, head_idx, :], TensorLayout((br, head_size), (num_heads * head_size, 1)), "global"
                )
                txgq = partition_src(tg_q, auto_copy())
                tg_do = tensor_view(
                    do_1[batch_idx, 0:, head_idx, :],
                    TensorLayout((head_size, seqlen_q), (1, num_heads * head_size)),
                    "global",
                )
                txgdo = partition_src(tg_do, auto_copy())

                tg_dp_sum = tensor_view(dp_sum[batch_idx, head_idx, 0:], TensorLayout((bc, seqlen_q), (0, 1)), "global")
                txgdpsum = partition_src(tg_dp_sum, auto_copy())

                copy(auto_copy((bc, head_size)), txgk, txsk)
                copy(auto_copy((br, head_size)), txgq, txsq)
                copy(auto_copy((head_size, br)), txgdo[:, :, 0], txsdo)

                cp_async_commit_group()

                tr_q = make_tensor("float16", layout_auto((br, inst_h * 2)), "register")
                tr_k = make_tensor("float16", layout_auto((bc, inst_h * 2)), "register")
                tr_v = make_tensor("float16", layout_auto((bc, head_size)), "register")
                txrq = partition_dst(tr_q, auto_copy())
                txrk = partition_dst(tr_k, auto_copy())
                txrv = partition_dst(tr_v, auto_copy())

                for hi in range(h_tile_max):
                    copy(auto_copy(), txgv[:, :, hi], txrv[:, :, hi])

                cp_async_wait_group(0)
                syncthreads()

                copy(auto_copy(), txSq[:, :, 0], txrq[:, :, 0])
                copy(auto_copy(), txSk[:, :, 0], txrk[:, :, 0])

                for i in grid(tr):
                    tr_qk = make_tensor("float32", auto_layout, "register")
                    fill(tr_qk, 0.0)
                    tr_dp_sum = make_tensor("float32", layout_auto((bc, br), (0, 1)), "register")
                    txrdpsum = partition_dst(tr_dp_sum, auto_copy())
                    copy(auto_copy((bc, br)), txgdpsum[:, :, i], txrdpsum)

                    if i >= 1:
                        copy(auto_copy(), txSq[:, :, 0], txrq[:, :, 0])

                    # compute k*q^T
                    for hi in grid(h_tile_max, "u+"):
                        h_tile_next = (hi + 1) % h_tile_max
                        if hi < h_tile_max - 1:
                            copy(auto_copy(), txSq[:, :, hi + 1], txrq[:, :, (hi + 1) % 2])
                        copy(auto_copy(), txSk[:, :, h_tile_next], txrk[:, :, (hi + 1) % 2])
                        mma(tiled_mma_qk, tr_qk, txrk[:, :, hi % 2], txrq[:, :, hi % 2], tr_qk)

                    li = make_tensor("float32", layout_auto((bc, br), (0, 1)), "register")
                    mi = make_tensor("float32", layout_auto((bc, br), (0, 1)), "register")
                    tg_l = tensor_view(l[batch_idx, head_idx, i * br :], TensorLayout((bc, br), (0, 1)), "global")
                    tg_m = tensor_view(m[batch_idx, head_idx, i * br :], TensorLayout((bc, br), (0, 1)), "global")
                    txgli = partition_src(tg_l, auto_copy())
                    txgmi = partition_src(tg_m, auto_copy())
                    txrli = partition_dst(li, auto_copy())
                    txrmi = partition_dst(mi, auto_copy())
                    copy(auto_copy((bc, br)), txgli, txrli)
                    copy(auto_copy((bc, br)), txgmi, txrmi)

                    p = exp(tr_qk - li)

                    # compute v*do^T
                    tr_dp = make_tensor("float32", auto_layout, "register")
                    fill(tr_dp, 0.0)

                    tr_dot = make_tensor("float16", layout_auto((br, inst_h * 2)), "register")
                    txrdot = partition_dst(tr_dot, auto_copy())
                    copy(auto_copy(), txSdot[:, :, 0], txrdot[:, :, 0])

                    for hi in grid(h_tile_max, "u+"):
                        h_tile_next = (hi + 1) % h_tile_max
                        if hi < h_tile_max - 1:
                            copy(auto_copy(), txSdot[:, :, h_tile_next], txrdot[:, :, (hi + 1) % 2])
                        # copy(auto_copy(), txSv[:, :, h_tile_next], txrv[:, :, (hi + 1) % 2])
                        mma(tiled_mma_qk, tr_dp, txrv[:, :, hi], txrdot[:, :, hi % 2], tr_dp)

                    ds = cast(p * (tr_dp - tr_dp_sum), f16)
                    txrds = partition_src(ds, auto_copy())
                    copy(auto_copy((bc, br)), txrds, txsds)

                    p_f16 = cast(p, f16)
                    txrp = partition_src(p_f16, auto_copy())
                    copy(auto_copy((bc, br)), txrp, txsp)

                    syncthreads()

                    # compute p*do
                    tr_p = make_tensor("float16", layout_auto((bc, inst_r * 2)), "register")
                    txrp1 = partition_dst(tr_p, auto_copy())
                    tr_do = make_tensor("float16", layout_auto((head_size, inst_r * 2)), "register")
                    txrdo = partition_dst(tr_do, auto_copy())

                    copy(auto_copy(), txSp[:, :, 0], txrp1[:, :, 0])
                    copy(auto_copy(), txSdo[:, :, 0], txrdo[:, :, 0])
                    for ri in grid(r_tile_max, "u+"):
                        if ri < r_tile_max - 1:
                            copy(auto_copy(), txSp[:, :, ri + 1], txrp1[:, :, (ri + 1) % 2])
                            copy(auto_copy(), txSdo[:, :, ri + 1], txrdo[:, :, (ri + 1) % 2])
                        mma(tiled_mma_pdo, tr_dv, txrp1[:, :, ri % 2], txrdo[:, :, ri % 2], tr_dv)

                    syncthreads()

                    if i < tr - 1:
                        copy(auto_copy((head_size, br)), txgdo[:, :, i + 1], txsdo)
                        cp_async_commit_group()

                    # compute ds*Q^T
                    tr_ds1 = make_tensor("float16", layout_auto((bc, inst_r * 2)), "register")
                    tr_qt = make_tensor("float16", layout_auto((head_size, inst_r * 2)), "register")
                    txrds1 = partition_dst(tr_ds1, auto_copy())
                    txrqt = partition_dst(tr_qt, auto_copy())
                    copy(auto_copy(), txSds[:, :, 0], txrds1[:, :, 0])
                    copy(auto_copy(), txSqt[:, :, 0], txrqt[:, :, 0])
                    for ri in grid(r_tile_max, "u+"):
                        if ri < r_tile_max - 1:
                            copy(auto_copy(), txSds[:, :, ri + 1], txrds1[:, :, (ri + 1) % 2])
                            copy(auto_copy(), txSqt[:, :, ri + 1], txrqt[:, :, (ri + 1) % 2])
                        mma(tiled_mma_dsq, tr_dk, txrds1[:, :, ri % 2], txrqt[:, :, ri % 2], tr_dk)

                    # compute ds^T*k^T
                    tr_dst = make_tensor("float16", layout_auto((br, inst_c * 2)), "register")
                    tr_kt = make_tensor("float16", layout_auto((head_size, inst_c * 2)), "register")
                    tr_dq = make_tensor("float32", auto_layout, "register")
                    fill(tr_dq, 0.0)
                    txrdst = partition_dst(tr_dst, auto_copy())
                    txrkt = partition_dst(tr_kt, auto_copy())
                    copy(auto_copy(), txSdst[:, :, 0], txrdst[:, :, 0])
                    copy(auto_copy(), txSkt[:, :, 0], txrkt[:, :, 0])

                    syncthreads()
                    # copy q
                    if i < tr - 1:
                        tg_q1 = tensor_view(
                            q[batch_idx, (i + 1) * br :, head_idx, :],
                            TensorLayout((br, head_size), (num_heads * head_size, 1)),
                            "global",
                        )
                        txgq1 = partition_src(tg_q1, auto_copy())
                        copy(auto_copy((br, head_size)), txgq1, txsq)
                        cp_async_commit_group()

                    for ci in grid(c_tile_max, "u+"):
                        if ci < c_tile_max - 1:
                            copy(auto_copy(), txSdst[:, :, ci + 1], txrdst[:, :, (ci + 1) % 2])
                            copy(auto_copy(), txSkt[:, :, ci + 1], txrkt[:, :, (ci + 1) % 2])
                        mma(tiled_mma_dsk, tr_dq, txrdst[:, :, ci % 2], txrkt[:, :, ci % 2], tr_dq)

                    tr_dq_f16 = cast(tr_dq, f16)
                    # tr_dq_contig = rearrange(tr_dq_f16, auto_layout, "register")
                    # dq = dq + tr_dq
                    tg_dq = tensor_view(
                        dq[batch_idx, i * br :, head_idx, :],
                        TensorLayout((br, head_size), (num_heads * head_size, 1)),
                        "global",
                    )
                    cute_atomic_add(tr_dq_f16, tg_dq)
                    cp_async_wait_group(0)
                    syncthreads()
                # write dk, dv
                tg_dK = tensor_view(
                    dk[batch_idx, seqk_part * seqk_partition + j * bc :, head_idx, :],
                    TensorLayout((bc, head_size), (num_heads_k * head_size, 1)),
                    "global",
                )
                tg_dV = tensor_view(
                    dv[batch_idx, seqk_part * seqk_partition + j * bc :, head_idx, :],
                    TensorLayout((bc, head_size), (num_heads_k * head_size, 1)),
                    "global",
                )
                txgdk = partition_dst(tg_dK, auto_copy())
                txgdv = partition_dst(tg_dV, auto_copy())

                tr_dk_f16 = cast(tr_dk, f16)
                tr_dv_f16 = cast(tr_dv, f16)
                tr_dK = rearrange(tr_dk_f16, auto_layout, "register")
                tr_dV = rearrange(tr_dv_f16, auto_layout, "register")
                txrdk = partition_src(tr_dK, auto_copy())
                txrdv = partition_src(tr_dV, auto_copy())
                copy(auto_copy((bc, head_size)), txrdk, txgdk)
                copy(auto_copy((bc, head_size)), txrdv, txgdv)

    func = script_module.build()
    # a_mem = hidet.empty([m, k], device="cuda")
    # b_mem = hidet.empty([k, n], device="cuda")
    # c_mem = hidet.empty([m, n], device="cuda")
    # func(a_mem, b_mem, c_mem)
    return func


def flash_attention_bwd_preprocess(batch_size, seqlen_q, num_heads, head_size):
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

    import sys

    float_max = f32.max_value

    bs = batch_size * num_heads
    br = 128
    tr = cdiv(seqlen_q, br)
    threads = 128

    with hidet.script_module() as script_module:

        @hidet.script
        def func(
            o: f16[batch_size, seqlen_q, num_heads, head_size],
            do_1: f16[batch_size, seqlen_q, num_heads, head_size],
            dp_sum: f32[batch_size, num_heads, seqlen_q],
            dq: f16[batch_size, seqlen_q, num_heads, head_size],
        ):
            attrs.func_kind = "cuda_kernel"
            attrs.cuda.block_dim = threads
            attrs.cuda.grid_dim = bs, tr
            attrs.cuda.dynamic_smem_bytes = 0

            pid_s = blockIdx.x
            pid_r = blockIdx.y
            batch_idx = pid_s // num_heads
            head_idx = pid_s % num_heads

            tg_o = tensor_view(
                o[batch_idx, pid_r * br :, head_idx, :],
                TensorLayout((br, head_size), (num_heads * head_size, 1)),
                "global",
            )
            txgo = partition_src(tg_o, auto_copy())
            tg_do = tensor_view(
                do_1[batch_idx, pid_r * br :, head_idx, :],
                TensorLayout((br, head_size), (num_heads * head_size, 1)),
                "global",
            )
            txgdo = partition_src(tg_do, auto_copy())

            tr_o = make_tensor("float16", layout_auto((br, head_size)), "register")
            tr_do = make_tensor("float16", layout_auto((br, head_size)), "register")
            txro = partition_dst(tr_o, auto_copy())
            txrdo = partition_dst(tr_do, auto_copy())
            copy(auto_copy((br, head_size)), txgo, txro)
            copy(auto_copy((br, head_size)), txgdo, txrdo)

            do_o = cast(tr_o, f32) * cast(tr_do, f32)
            tr_dp_sum = reduce_sum(do_o, 1)
            txrdp_sum = partition_src(tr_dp_sum, auto_copy())
            tg_dp_sum = tensor_view(
                dp_sum[batch_idx, head_idx, pid_r * br :], TensorLayout((br, head_size), (1, 0)), "global"
            )
            txgdp_sum = partition_dst(tg_dp_sum, auto_copy())
            copy(auto_copy((br, head_size)), txrdp_sum, txgdp_sum)

            tg_dq = tensor_view(
                dq[batch_idx, pid_r * br :, head_idx, :],
                TensorLayout((br, head_size), (num_heads * head_size, 1)),
                "global",
            )
            txgdq = partition_src(tg_dq, auto_copy())
            tr_dq = make_tensor("float16", layout_auto((br, head_size)), "register")
            txrdq = partition_dst(tr_dq, auto_copy())
            fill(tr_dq, 0.0)
            copy(auto_copy((br, head_size)), txrdq, txgdq)

    func = script_module.build()
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
    # shrink the range of random inputs, so that the computation with fp16 for
    # bwd won't overflow or underflow
    lo = -2
    hi = 2
    q = torch.randint(low=lo, high=hi, size=(batch_size, seqlen_q, num_heads, head_size), dtype=dtype, device=device)
    k = torch.randint(low=lo, high=hi, size=(batch_size, seqlen_k, num_heads_k, head_size), dtype=dtype, device=device)
    v = torch.randint(low=lo, high=hi, size=(batch_size, seqlen_k, num_heads_k, head_size), dtype=dtype, device=device)
    o = torch.randint(low=lo, high=hi, size=(batch_size, seqlen_q, num_heads, head_size), dtype=dtype, device=device)
    do = torch.randint(low=0, high=hi, size=(batch_size, seqlen_q, num_heads, head_size), dtype=dtype, device=device)
    lse = torch.randint(
        low=5 + lo, high=5 + hi, size=(batch_size, num_heads, seqlen_q), dtype=torch.float32, device=device
    )
    m = torch.randint(
        low=5 + lo, high=5 + hi, size=(batch_size, num_heads, seqlen_q), dtype=torch.float32, device=device
    )

    dq = torch.empty((batch_size, seqlen_q, num_heads, head_size), dtype=dtype, device=device)
    dk = torch.empty((batch_size, seqlen_k, num_heads_k, head_size), dtype=dtype, device=device)
    dv = torch.empty((batch_size, seqlen_k, num_heads_k, head_size), dtype=dtype, device=device)

    import numpy as np

    q = q / np.sqrt(head_size)
    k = k / np.sqrt(head_size)
    o = o / head_size

    if return_hidet:
        q = hidet.from_torch(q)
        k = hidet.from_torch(k)
        v = hidet.from_torch(v)
        o = hidet.from_torch(o)
        do = hidet.from_torch(do)

        dq = hidet.from_torch(dq)
        dk = hidet.from_torch(dk)
        dv = hidet.from_torch(dv)
        lse = hidet.from_torch(lse)
        m = hidet.from_torch(m)

    return q, k, v, o, do, dq, dk, dv, lse, m


# we just test the compilation here, but not the correctness of the results.
@pytest.mark.parametrize(
    "batch_size,num_heads,num_heads_k,head_size,seqlen_q,seqlen_k,num_parallel_seqk_parts",
    [(1, 16, 16, 128, 4096, 4096, 8), (1, 16, 16, 128, 2048, 2048, 2), (1, 16, 16, 128, 1024, 1024, 4)],
)
def test_v3(batch_size, seqlen_q, num_heads, head_size, seqlen_k, num_heads_k, num_parallel_seqk_parts):
    flash_attention_v3_bwd(batch_size, seqlen_q, num_heads, head_size, seqlen_k, num_heads_k, num_parallel_seqk_parts)


@pytest.mark.parametrize(
    "batch_size,num_heads,num_heads_k,head_size,seqlen_q,seqlen_k,num_parallel_seqk_parts",
    [(1, 16, 16, 128, 4096, 4096, 8), (1, 16, 16, 128, 2048, 2048, 2), (1, 16, 16, 128, 1024, 1024, 4)],
)
def test_v5(batch_size, seqlen_q, num_heads, head_size, seqlen_k, num_heads_k, num_parallel_seqk_parts):
    flash_attention_v3_bwd(batch_size, seqlen_q, num_heads, head_size, seqlen_k, num_heads_k, num_parallel_seqk_parts)


@pytest.mark.parametrize(
    "batch_size,num_heads,num_heads_k,head_size,seqlen_q,seqlen_k,num_parallel_seqk_parts",
    [
        (1, 16, 16, 128, 4096, 4096, 8),
        (1, 16, 16, 128, 2048, 2048, 2),
        (1, 16, 16, 128, 1024, 1024, 4),
        (1, 16, 16, 64, 4096, 4096, 8),
        (1, 16, 16, 64, 2048, 2048, 2),
        (1, 16, 16, 64, 1024, 1024, 4),
    ],
)
def test_v4(batch_size, seqlen_q, num_heads, head_size, seqlen_k, num_heads_k, num_parallel_seqk_parts):
    func_preprocess = flash_attention_bwd_preprocess(batch_size, seqlen_q, num_heads, head_size)
    func = flash_attention_v4_bwd(
        batch_size, seqlen_q, num_heads, head_size, seqlen_k, num_heads_k, num_parallel_seqk_parts
    )
    q, k, v, o, do, dq, dk, dv, lse, m = data(batch_size, seqlen_q, num_heads, head_size, seqlen_k, num_heads_k)
    dq_seqk_parallel_parts = torch.empty(
        (batch_size, num_parallel_seqk_parts, seqlen_q, num_heads, head_size), dtype=torch.float16, device="cuda"
    )
    lock = torch.empty((batch_size, num_heads), dtype=torch.int32, device="cuda")
    dq_seqk_parallel_parts = hidet.from_torch(dq_seqk_parallel_parts)
    dp_sum = torch.empty((batch_size, num_heads, seqlen_q), dtype=torch.float32, device="cuda")
    lock = hidet.from_torch(lock)
    mean, min_lat, max_lat = bench(func, (q, k, v, o, do, dq, dk, dv, lse, m, dq_seqk_parallel_parts, lock, dp_sum))
    flops = (
        2.5
        * 2.0
        * (
            batch_size * seqlen_q * num_heads * seqlen_k * head_size
            + batch_size * seqlen_q * num_heads_k * seqlen_k * head_size
        )
    )
    from hidet.ir.dtypes import f16

    func_preprocess(o, do, dp_sum, dq)

    memory = f16.nbytes * (
        batch_size * num_heads * seqlen_q * head_size
        + batch_size * num_heads_k * seqlen_k * head_size
        + batch_size * num_heads_k * seqlen_k * head_size
        + batch_size * seqlen_q * num_heads * head_size
    )
    print("time={:.3f} ms, performance={:.3f} TFLOPS".format(mean, flops / (1e9 * mean)))
    print("time={:.3f} ms, bandwidth={:.3f} GB/s".format(mean, memory / (1e6 * mean)))

    def fn():
        func_preprocess(o, do, dp_sum, dq)
        func(q, k, v, o, do, dq, dk, dv, lse, m, dq_seqk_parallel_parts, lock, dp_sum)

    from hidet.utils.benchmark import do_bench

    mean, min_lat, max_lat = bench(fn, ())
    print("time={:.3f} ms, performance={:.3f} TFLOPS".format(mean, flops / (1e9 * mean)))

    dq1 = torch.empty((batch_size, seqlen_q, num_heads, head_size), dtype=torch.float16, device="cuda")
    dk1 = torch.empty((batch_size, seqlen_k, num_heads_k, head_size), dtype=torch.float16, device="cuda")
    dv1 = torch.empty((batch_size, seqlen_k, num_heads_k, head_size), dtype=torch.float16, device="cuda")

    try:
        import flash_attn_2_cuda as flash_attn_cuda
    except ImportError:
        pytest.skip('skip the accuracy test since the flash_attn library is not installed')

    def fn2(dq1, dk1, dv1):
        dq1, dk1, dv1, softmax_d = flash_attn_cuda.bwd(
            do, q, k, v, o, lse, dq1, dk1, dv1, None, 0.0, 1.0, False, -1, -1, False, None, None
        )
        return dq1, dk1, dv1

    dq1, dk1, dv1 = fn2(dq1, dk1, dv1)

    mean, min_lat, max_lat = bench(fn2, (dq1, dk1, dv1))
    print("time={:.3f} ms, performance={:.3f} TFLOPS".format(mean, flops / (1e9 * mean)))

    import numpy as np

    np.set_printoptions(threshold=3000, linewidth=200, edgeitems=100)
    np.testing.assert_allclose(actual=dv1.cpu().numpy(), desired=dv.cpu().numpy(), rtol=1e-2)
    np.testing.assert_allclose(actual=dk1.cpu().numpy(), desired=dk.cpu().numpy(), rtol=5e-2)
    np.testing.assert_allclose(actual=dq1.cpu().numpy(), desired=dq.cpu().numpy(), rtol=5e-2)
