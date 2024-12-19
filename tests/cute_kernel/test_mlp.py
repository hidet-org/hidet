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
    silu,
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


def mlp(batch_size, seqlen, dmodel, dffn):
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
    tiled_mma_xw1 = TiledMma(mma_atom, [warp_in_threadblock])

    print(tiled_mma_xw1.str_indented())
    x_shape, x_tv_layout = tiled_mma_xw1.a_tv_layout()
    w1_shape, w1_tv_layout = tiled_mma_xw1.b_tv_layout()
    xw1_shape, xw1_tv_layout = tiled_mma_xw1.c_tv_layout()
    print(x_shape, x_tv_layout)
    print(w1_shape, w1_tv_layout)
    print(xw1_shape, xw1_tv_layout)

    xw1_t, xw1_v = canonicalize(xw1_tv_layout)

    threads = xw1_t.size()

    bm, inst_k = x_shape
    bn, inst_k_ = w1_shape
    bm_, bn_ = xw1_shape
    assert bm == bm_ and bn == bn_ and inst_k == inst_k_

    a = TensorLayout(((4, 8), (2, 2)), ((16, 1), (8, 64)))
    b = TensorLayout(((4, 8), (2, 2, 2)), ((32, 1), (16, 8, 128)))
    c = TensorLayout(((4, 8), (2, 2)), ((2, 8), (1, 64)))
    mma_atom = MmaAtom("warp", (8, 16, 16), a, b, c, c, (1, 1))
    warp_in_threadblock = Level("warp", "thread_block", (1, 4), TensorLayout((1, 4)), (1, 2))
    tiled_mma_yw2 = TiledMma(mma_atom, [warp_in_threadblock])

    print(tiled_mma_yw2.str_indented())
    y_shape, y_tv_layout = tiled_mma_yw2.a_tv_layout()
    w2_shape, _ = tiled_mma_yw2.b_tv_layout()

    _, inst_n = y_shape
    bl, _ = w2_shape

    from hidet.ir.cute.algorithm import auto_copy, auto_mma
    from hidet.ir.cute import auto_layout, layout_auto

    from hidet.utils.py import cdiv
    from hidet.lang import grid

    blocks = cdiv(dffn, bn)
    bk = 32
    stages = 2

    with hidet.script_module() as script_module:

        @hidet.script
        def func(
            x: f16[batch_size * seqlen, dmodel],
            w1: f16[dmodel, dffn],
            w2: f16[dffn, dmodel],
            z: f16[batch_size * seqlen, dmodel],
            y: f16[batch_size * seqlen, dffn],
        ):
            attrs.func_kind = "cuda_kernel"
            attrs.cuda.block_dim = threads
            attrs.cuda.grid_dim = blocks, 1
            attrs.cuda.dynamic_smem_bytes = 0

            pid = blockIdx.x

            tr_x = make_tensor("float16", layout_auto((bm, inst_k * 2)), "register")
            tr_w1 = make_tensor("float16", layout_auto((bn, inst_k * 2)), "register")
            tr_xw1 = make_tensor("float32", auto_layout, "register")
            fill(tr_xw1, 0.0)

            ts_x = make_tensor("float16", TensorLayout((bm, bk), (bk, 1)), "shared")
            txsx = partition_dst(ts_x, auto_copy())
            txSx = partition_src(ts_x, auto_copy())

            ts_w1 = make_tensor("float16", TensorLayout((bn, bk), (1, bn)), "shared")
            txsw1 = partition_dst(ts_w1, auto_copy())
            txSw1 = partition_src(ts_w1, auto_copy())

            ts_y = make_tensor("float16", layout_auto((bm, bn)), "shared")
            txSy = partition_src(ts_y, auto_copy())

            tg_x = tensor_view(x[:, :], TensorLayout((bm, dmodel), (dmodel, 1)), "global")
            txgx = partition_src(tg_x, auto_copy())

            tg_w1 = tensor_view(w1[:, pid * bn :], TensorLayout((bn, dmodel), (1, dffn)), "global")
            txgw1 = partition_src(tg_w1, auto_copy())

            ts_w2 = make_tensor("float16", TensorLayout((bl, bn, stages), (1, bl, bn * bl)), "shared")
            txsw2 = partition_dst(ts_w2, auto_copy())
            txSw2 = partition_src(ts_w2, auto_copy())

            for s in range(stages - 1):
                tg_w2 = tensor_view(w2[pid * bn :, s * bl :], TensorLayout((bl, bn), (1, dmodel)), "global")
                txgw2 = partition_src(tg_w2, auto_copy())
                copy(auto_copy((bl, bn)), txgw2, txsw2[:, :, s])

            txrx = partition_dst(tr_x, auto_copy())
            txrw1 = partition_dst(tr_w1, auto_copy())

            do1 = (dmodel + bk - 1) // bk
            do2 = (dmodel + bl - 1) // bl
            for i in range(do1):
                copy(auto_copy((bm, bk)), txgx[:, :, i], txsx)
                copy(auto_copy((bn, bk)), txgw1[:, :, i], txsw1)
                cp_async_wait_all()
                syncthreads()

                copy(auto_copy(), txSx[:, :, 0], txrx[:, :, 0])
                copy(auto_copy(), txSw1[:, :, 0], txrw1[:, :, 0])
                di1 = (bk + inst_k - 1) // inst_k
                for j in grid(di1, "u+"):
                    if j < di1 - 1:
                        copy(auto_copy(), txSx[:, :, j + 1], txrx[:, :, (j + 1) % 2])
                        copy(auto_copy(), txSw1[:, :, j + 1], txrw1[:, :, (j + 1) % 2])
                    mma(tiled_mma_xw1, tr_xw1, txrx[:, :, j % 2], txrw1[:, :, j % 2], tr_xw1)
                syncthreads()

            tg_y = tensor_view(y[:, pid * bn :], TensorLayout((bm, bn), (dffn, 1)), "global")
            tr_y1 = rearrange(cast(tr_xw1, f16), auto_layout, "register")
            txry = partition_src(tr_y1, auto_copy())
            txgy = partition_dst(tg_y, auto_copy())
            copy(auto_copy((bm, bn)), txry, txgy)

            tr_y = make_tensor("float16", layout_auto((bm, inst_n * 2)), "register")
            tr_w2 = make_tensor("float16", layout_auto((bl, inst_n * 2)), "register")
            tr_yw2 = make_tensor("float32", auto_layout, "register")

            syncthreads()
            smem_pipe_read = 0
            smem_pipe_write = stages - 1

            txry1 = partition_dst(tr_y, auto_copy())
            txrw2 = partition_dst(tr_w2, auto_copy())

            copy(auto_copy(), txSy[:, :, 0], txry1[:, :, 0])
            copy(auto_copy(), txSw2[:, :, 0, smem_pipe_read], txrw2[:, :, 0])

            di2 = (bn + inst_n - 1) // inst_n
            for j in range(do2):
                fill(tr_yw2, 0.0)

                for i in grid(di2, "u+"):
                    if i == di2 - 1:
                        cp_async_wait_group(allow_on_fly_groups=0)
                        syncthreads()

                    i_tile_next = (i + 1) % di2
                    copy(auto_copy(), txSy[:, :, i_tile_next], txry1[:, :, (i + 1) % 2])
                    copy(auto_copy(), txSw2[:, :, i_tile_next, smem_pipe_read], txrw2[:, :, (i + 1) % 2])
                    if i == 0:
                        if j + stages - 1 < do2:
                            tg_w2 = tensor_view(
                                w2[pid * bn :, (j + stages - 1) * bl :], TensorLayout((bl, bn), (1, dmodel)), "global"
                            )
                            txgw2 = partition_src(tg_w2, auto_copy())
                            copy(auto_copy((bl, bn)), txgw2, txsw2[:, :, smem_pipe_write])
                        smem_pipe_write = smem_pipe_read
                        cp_async_commit_group()

                    if i == di2 - 2:
                        smem_pipe_read += 1
                        smem_pipe_read = 0 if smem_pipe_read == stages else smem_pipe_read

                    mma(tiled_mma_yw2, tr_yw2, txry1[:, :, i % 2], txrw2[:, :, i % 2], tr_yw2)
                tr_z = cast(tr_yw2, f16)
                tg_z = tensor_view(z[:, j * bl :], TensorLayout((bm, bl), (dmodel, 1)), "global")
                # txrz = partition_src(tr_z, auto_copy())
                # txgz = partition_dst(tg_z, auto_copy())
                # copy(auto_copy((bm, bl)), txrz, txgz)
                cute_atomic_add(tr_z, tg_z)

    func = script_module.build()
    # a_mem = hidet.empty([m, k], device="cuda")
    # b_mem = hidet.empty([k, n], device="cuda")
    # c_mem = hidet.empty([m, n], device="cuda")
    # func(a_mem, b_mem, c_mem)
    return func


def mlp_v2(batch_size, seqlen, dmodel, dffn):
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
    tiled_mma_xw1 = TiledMma(mma_atom, [warp_in_threadblock])

    print(tiled_mma_xw1.str_indented())
    x_shape, x_tv_layout = tiled_mma_xw1.a_tv_layout()
    w1_shape, w1_tv_layout = tiled_mma_xw1.b_tv_layout()
    xw1_shape, xw1_tv_layout = tiled_mma_xw1.c_tv_layout()
    print(x_shape, x_tv_layout)
    print(w1_shape, w1_tv_layout)
    print(xw1_shape, xw1_tv_layout)

    xw1_t, xw1_v = canonicalize(xw1_tv_layout)

    threads = xw1_t.size()

    bm, inst_k = x_shape
    bn, inst_k_ = w1_shape
    bm_, bn_ = xw1_shape
    assert bm == bm_ and bn == bn_ and inst_k == inst_k_

    a = TensorLayout(((4, 8), (2, 2)), ((16, 1), (8, 64)))
    b = TensorLayout(((4, 8), (2, 2, 2)), ((32, 1), (16, 8, 128)))
    c = TensorLayout(((4, 8), (2, 2)), ((2, 8), (1, 64)))
    mma_atom = MmaAtom("warp", (8, 16, 16), a, b, c, c, (1, 1))
    warp_in_threadblock = Level("warp", "thread_block", (1, 4), TensorLayout((1, 4)), (1, 2))
    tiled_mma_yw2 = TiledMma(mma_atom, [warp_in_threadblock])

    print(tiled_mma_yw2.str_indented())
    y_shape, y_tv_layout = tiled_mma_yw2.a_tv_layout()
    w2_shape, _ = tiled_mma_yw2.b_tv_layout()

    _, inst_n = y_shape
    bl, _ = w2_shape

    from hidet.ir.cute.algorithm import auto_copy, auto_mma
    from hidet.ir.cute import auto_layout, layout_auto

    from hidet.utils.py import cdiv
    from hidet.lang import grid

    blocks = cdiv(dffn, bn)
    blocks1 = cdiv(dmodel, bl)
    bk = 32
    stages = 2
    n_parallel_parts = blocks // blocks1
    n_partition = bn
    while n_partition * n_parallel_parts < dffn:
        n_partition += bn

    with hidet.script_module() as script_module:

        @hidet.script
        def func(
            x: f16[batch_size * seqlen, dmodel],
            w1: f16[dmodel, dffn],
            w2: f16[dffn, dmodel],
            z: f16[batch_size * seqlen, dmodel],
            y: f16[batch_size * seqlen, dffn],
            lock: i32[blocks],
        ):
            attrs.func_kind = "cuda_kernel"
            attrs.cuda.block_dim = threads
            attrs.cuda.grid_dim = blocks, 1
            attrs.cuda.dynamic_smem_bytes = 0

            pid = blockIdx.x

            if threadIdx.x == 0:
                lock[pid] = 0

            tr_x = make_tensor("float16", layout_auto((bm, inst_k * 2)), "register")
            tr_w1 = make_tensor("float16", layout_auto((bn, inst_k * 2)), "register")
            tr_xw1 = make_tensor("float32", auto_layout, "register")
            fill(tr_xw1, 0.0)

            ts_x = make_tensor("float16", TensorLayout((bm, bk), (bk, 1)), "shared")
            txsx = partition_dst(ts_x, auto_copy())
            txSx = partition_src(ts_x, auto_copy())

            ts_w1 = make_tensor("float16", TensorLayout((bn, bk), (1, bn)), "shared")
            txsw1 = partition_dst(ts_w1, auto_copy())
            txSw1 = partition_src(ts_w1, auto_copy())

            ts_y = make_tensor("float16", layout_auto((bm, bn, stages)), "shared")
            txsy = partition_dst(ts_y, auto_copy())
            txSy = partition_src(ts_y, auto_copy())

            tg_x = tensor_view(x[:, :], TensorLayout((bm, dmodel), (dmodel, 1)), "global")
            txgx = partition_src(tg_x, auto_copy())

            tg_w1 = tensor_view(w1[:, pid * bn :], TensorLayout((bn, dmodel), (1, dffn)), "global")
            txgw1 = partition_src(tg_w1, auto_copy())

            ts_w2 = make_tensor("float16", TensorLayout((bl, bn, stages), (1, bl, bn * bl)), "shared")
            txsw2 = partition_dst(ts_w2, auto_copy())
            txSw2 = partition_src(ts_w2, auto_copy())

            n_part = pid % n_parallel_parts
            pid_l = pid // n_parallel_parts

            tg_w2 = tensor_view(
                w2[n_part * n_partition :, pid_l * bl :], TensorLayout((bl, n_partition), (1, dmodel)), "global"
            )
            txgw2 = partition_src(tg_w2, auto_copy())
            for s in range(stages - 1):
                copy(auto_copy((bl, bn)), txgw2[:, :, s], txsw2[:, :, s])

            txrx = partition_dst(tr_x, auto_copy())
            txrw1 = partition_dst(tr_w1, auto_copy())

            do1 = (dmodel + bk - 1) // bk
            for i in range(do1):
                copy(auto_copy((bm, bk)), txgx[:, :, i], txsx)
                copy(auto_copy((bn, bk)), txgw1[:, :, i], txsw1)
                cp_async_wait_all()
                syncthreads()

                copy(auto_copy(), txSx[:, :, 0], txrx[:, :, 0])
                copy(auto_copy(), txSw1[:, :, 0], txrw1[:, :, 0])
                di1 = (bk + inst_k - 1) // inst_k
                for j in grid(di1, "u+"):
                    if j < di1 - 1:
                        copy(auto_copy(), txSx[:, :, j + 1], txrx[:, :, (j + 1) % 2])
                        copy(auto_copy(), txSw1[:, :, j + 1], txrw1[:, :, (j + 1) % 2])
                    mma(tiled_mma_xw1, tr_xw1, txrx[:, :, j % 2], txrw1[:, :, j % 2], tr_xw1)
                syncthreads()

            tg_y = tensor_view(y[:, pid * bn :], TensorLayout((bm, bn), (dffn, 1)), "global")
            tr_y1 = rearrange(cast(tr_xw1, f16), auto_layout, "register")
            txry = partition_src(tr_y1, auto_copy())
            txgy = partition_dst(tg_y, auto_copy())
            copy(auto_copy((bm, bn)), txry, txgy)
            counter = ~lock[pid]
            release_seq_semaphore(counter, 1)

            tr_y = make_tensor("float16", layout_auto((bm, inst_n * 2)), "register")
            tr_w2 = make_tensor("float16", layout_auto((bl, inst_n * 2)), "register")
            tr_yw2 = make_tensor("float32", auto_layout, "register")
            fill(tr_yw2, 0.0)

            tg_y1 = tensor_view(y[:, n_part * n_partition :], TensorLayout((bm, n_partition), (dffn, 1)), "global")
            txgy1 = partition_src(tg_y1, auto_copy())
            wait_id = n_part * n_partition // bn
            wait_counter = ~lock[wait_id]
            for s in range(stages - 1):
                acquire_seq_semaphore(wait_counter + s, 1)
                copy(auto_copy((bm, bn)), txgy1[:, :, s], txsy[:, :, s])
                cp_async_commit_group()

            syncthreads()
            smem_pipe_read = 0
            smem_pipe_write = stages - 1

            txry1 = partition_dst(tr_y, auto_copy())
            txrw2 = partition_dst(tr_w2, auto_copy())

            copy(auto_copy(), txSy[:, :, 0, smem_pipe_read], txry1[:, :, 0])
            copy(auto_copy(), txSw2[:, :, 0, smem_pipe_read], txrw2[:, :, 0])

            do2 = (n_partition + bn - 1) // bn
            di2 = (bn + inst_n - 1) // inst_n
            for j in range(do2):
                for i in grid(di2, "u+"):
                    if i == di2 - 1:
                        cp_async_wait_group(allow_on_fly_groups=0)
                        syncthreads()

                    i_tile_next = (i + 1) % di2
                    copy(auto_copy(), txSy[:, :, i_tile_next, smem_pipe_read], txry1[:, :, (i + 1) % 2])
                    copy(auto_copy(), txSw2[:, :, i_tile_next, smem_pipe_read], txrw2[:, :, (i + 1) % 2])
                    if i == 0:
                        if j + stages - 1 < do2:
                            copy(auto_copy((bl, bn)), txgw2[:, :, j + stages - 1], txsw2[:, :, smem_pipe_write])
                            acquire_seq_semaphore(wait_counter + j + stages - 1, 1)
                            copy(auto_copy((bm, bn)), txgy1[:, :, j + stages - 1], txsy[:, :, smem_pipe_write])
                        smem_pipe_write = smem_pipe_read
                        cp_async_commit_group()

                    if i == di2 - 2:
                        smem_pipe_read += 1
                        smem_pipe_read = 0 if smem_pipe_read == stages else smem_pipe_read

                    mma(tiled_mma_yw2, tr_yw2, txry1[:, :, i % 2], txrw2[:, :, i % 2], tr_yw2)
            tr_z = cast(tr_yw2, f16)
            tg_z = tensor_view(z[:, pid_l * bl :], TensorLayout((bm, bl), (dmodel, 1)), "global")
            cute_atomic_add(tr_z, tg_z)

    func = script_module.build()
    # a_mem = hidet.empty([m, k], device="cuda")
    # b_mem = hidet.empty([k, n], device="cuda")
    # c_mem = hidet.empty([m, n], device="cuda")
    # func(a_mem, b_mem, c_mem)
    return func


def mlp_v3(batch_size, seqlen, dmodel, dffn):
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
    tiled_mma_xw1 = TiledMma(mma_atom, [warp_in_threadblock])

    print(tiled_mma_xw1.str_indented())
    x_shape, x_tv_layout = tiled_mma_xw1.a_tv_layout()
    w1_shape, w1_tv_layout = tiled_mma_xw1.b_tv_layout()
    xw1_shape, xw1_tv_layout = tiled_mma_xw1.c_tv_layout()
    print(x_shape, x_tv_layout)
    print(w1_shape, w1_tv_layout)
    print(xw1_shape, xw1_tv_layout)

    xw1_t, xw1_v = canonicalize(xw1_tv_layout)

    threads = xw1_t.size()

    bm, inst_k = x_shape
    bn, inst_k_ = w1_shape
    bm_, bn_ = xw1_shape
    assert bm == bm_ and bn == bn_ and inst_k == inst_k_

    a = TensorLayout(((4, 8), (2, 2)), ((16, 1), (8, 64)))
    b = TensorLayout(((4, 8), (2, 2, 2)), ((32, 1), (16, 8, 128)))
    c = TensorLayout(((4, 8), (2, 2)), ((2, 8), (1, 64)))
    mma_atom = MmaAtom("warp", (8, 16, 16), a, b, c, c, (1, 1))
    warp_in_threadblock = Level("warp", "thread_block", (1, 4), TensorLayout((1, 4)), (1, 2))
    tiled_mma_yw2 = TiledMma(mma_atom, [warp_in_threadblock])

    print(tiled_mma_yw2.str_indented())
    y_shape, y_tv_layout = tiled_mma_yw2.a_tv_layout()
    w2_shape, _ = tiled_mma_yw2.b_tv_layout()

    _, inst_n = y_shape
    bl, _ = w2_shape

    from hidet.ir.cute.algorithm import auto_copy, auto_mma
    from hidet.ir.cute import auto_layout, layout_auto

    from hidet.utils.py import cdiv
    from hidet.lang import grid

    blocks = cdiv(dffn, bn)
    blocks1 = cdiv(dmodel, bl)
    bk = 64
    stages = 2
    n_parallel_parts = blocks // blocks1
    n_partition = bn
    while n_partition * n_parallel_parts < dffn:
        n_partition += bn

    with hidet.script_module() as script_module:

        @hidet.script
        def func(
            x: f16[batch_size * seqlen, dmodel],
            w1: f16[dmodel, dffn * 2],
            # w2: f16[dmodel, dffn],
            w2: f16[dffn, dmodel],
            z: f16[batch_size * seqlen, dmodel],
            y: f16[batch_size * seqlen, dffn],
            lock: i32[blocks],
        ):
            attrs.func_kind = "cuda_kernel"
            attrs.cuda.block_dim = threads
            attrs.cuda.grid_dim = blocks, 1
            attrs.cuda.dynamic_smem_bytes = 0

            pid = blockIdx.x

            if threadIdx.x == 0:
                lock[pid] = 0

            tr_x = make_tensor("float16", layout_auto((bm, inst_k * 2)), "register")
            tr_w1 = make_tensor("float16", layout_auto((bn, inst_k * 2)), "register")
            tr_xw1 = make_tensor("float32", auto_layout, "register")
            tr_w1_1 = make_tensor("float16", layout_auto((bn, inst_k * 2)), "register")
            tr_xw1_1 = make_tensor("float32", auto_layout, "register")
            fill(tr_xw1, 0.0)
            fill(tr_xw1_1, 0.0)

            ts_x = make_tensor("float16", TensorLayout((bm, bk), (bk, 1)), "shared")
            txsx = partition_dst(ts_x, auto_copy())
            txSx = partition_src(ts_x, auto_copy())

            ts_w1 = make_tensor("float16", TensorLayout((bn, bk), (1, bn)), "shared")
            txsw1 = partition_dst(ts_w1, auto_copy())
            txSw1 = partition_src(ts_w1, auto_copy())

            ts_w1_1 = make_tensor("float16", TensorLayout((bn, bk), (1, bn)), "shared")
            txsw1_1 = partition_dst(ts_w1_1, auto_copy())
            txSw1_1 = partition_src(ts_w1_1, auto_copy())

            ts_y = make_tensor("float16", layout_auto((bm, bn, stages)), "shared")
            txsy = partition_dst(ts_y, auto_copy())
            txSy = partition_src(ts_y, auto_copy())

            tg_x = tensor_view(x[:, :], TensorLayout((bm, dmodel), (dmodel, 1)), "global")
            txgx = partition_src(tg_x, auto_copy())

            tg_w1 = tensor_view(w1[:, pid * bn :], TensorLayout((bn, dmodel), (1, 2 * dffn)), "global")
            txgw1 = partition_src(tg_w1, auto_copy())

            tg_w1_1 = tensor_view(w1[:, dffn + pid * bn :], TensorLayout((bn, dmodel), (1, 2 * dffn)), "global")
            txgw1_1 = partition_src(tg_w1_1, auto_copy())

            ts_w2 = make_tensor("float16", TensorLayout((bl, bn, stages), (1, bl, bn * bl)), "shared")
            txsw2 = partition_dst(ts_w2, auto_copy())
            txSw2 = partition_src(ts_w2, auto_copy())

            n_part = pid % n_parallel_parts
            pid_l = pid // n_parallel_parts

            tg_w2 = tensor_view(
                w2[n_part * n_partition :, pid_l * bl :], TensorLayout((bl, n_partition), (1, dmodel)), "global"
            )
            txgw2 = partition_src(tg_w2, auto_copy())
            tr_w2_1 = make_tensor("float16", layout_auto((bl, bn * stages)), "register")
            txrw2_1 = partition_dst(tr_w2_1, auto_copy())
            for s in range(stages - 1):
                copy(auto_copy((bl, bn)), txgw2[:, :, s], txrw2_1[:, :, s])
            txsw2_1 = partition_dst(ts_w2, auto_copy())

            txrx = partition_dst(tr_x, auto_copy())
            txrw1 = partition_dst(tr_w1, auto_copy())
            txrw1_1 = partition_dst(tr_w1_1, auto_copy())

            do1 = (dmodel + bk - 1) // bk
            for i in range(do1):
                copy(auto_copy((bm, bk)), txgx[:, :, i], txsx)
                copy(auto_copy((bn, bk)), txgw1[:, :, i], txsw1)
                copy(auto_copy((bn, bk)), txgw1_1[:, :, i], txsw1_1)
                cp_async_wait_all()
                syncthreads()

                copy(auto_copy(), txSx[:, :, 0], txrx[:, :, 0])
                copy(auto_copy(), txSw1[:, :, 0], txrw1[:, :, 0])
                copy(auto_copy(), txSw1_1[:, :, 0], txrw1_1[:, :, 0])
                di1 = (bk + inst_k - 1) // inst_k
                for j in grid(di1, "u+"):
                    if j < di1 - 1:
                        copy(auto_copy(), txSx[:, :, j + 1], txrx[:, :, (j + 1) % 2])
                        copy(auto_copy(), txSw1[:, :, j + 1], txrw1[:, :, (j + 1) % 2])
                        copy(auto_copy(), txSw1_1[:, :, j + 1], txrw1_1[:, :, (j + 1) % 2])
                    mma(tiled_mma_xw1, tr_xw1, txrx[:, :, j % 2], txrw1[:, :, j % 2], tr_xw1)
                    mma(tiled_mma_xw1, tr_xw1_1, txrx[:, :, j % 2], txrw1_1[:, :, j % 2], tr_xw1_1)
                syncthreads()

            for s in range(stages - 1):
                copy(auto_copy((bl, bn)), txrw2_1[:, :, s], txsw2_1[:, :, s])
            tg_y = tensor_view(y[:, pid * bn :], TensorLayout((bm, bn), (dffn, 1)), "global")
            tr_y1 = rearrange(cast(silu(tr_xw1) * tr_xw1_1, f16), auto_layout, "register")
            txry = partition_src(tr_y1, auto_copy())
            txgy = partition_dst(tg_y, auto_copy())
            copy(auto_copy((bm, bn)), txry, txgy)
            counter = ~lock[pid]
            release_seq_semaphore(counter, 1)

            tr_y = make_tensor("float16", layout_auto((bm, inst_n * 2)), "register")
            tr_w2 = make_tensor("float16", layout_auto((bl, inst_n * 2)), "register")
            tr_yw2 = make_tensor("float32", auto_layout, "register")
            fill(tr_yw2, 0.0)

            tg_y1 = tensor_view(y[:, n_part * n_partition :], TensorLayout((bm, n_partition), (dffn, 1)), "global")
            txgy1 = partition_src(tg_y1, auto_copy())
            wait_id = n_part * n_partition // bn
            wait_counter = ~lock[wait_id]
            for s in range(stages - 1):
                acquire_seq_semaphore(wait_counter + s, 1)
                copy(auto_copy((bm, bn)), txgy1[:, :, s], txsy[:, :, s])
                cp_async_commit_group()

            syncthreads()
            smem_pipe_read = 0
            smem_pipe_write = stages - 1

            txry1 = partition_dst(tr_y, auto_copy())
            txrw2 = partition_dst(tr_w2, auto_copy())

            copy(auto_copy(), txSy[:, :, 0, smem_pipe_read], txry1[:, :, 0])
            copy(auto_copy(), txSw2[:, :, 0, smem_pipe_read], txrw2[:, :, 0])

            do2 = (n_partition + bn - 1) // bn
            di2 = (bn + inst_n - 1) // inst_n
            for j in range(do2):
                for i in grid(di2, "u+"):
                    if i == di2 - 1:
                        cp_async_wait_group(allow_on_fly_groups=0)
                        syncthreads()

                    i_tile_next = (i + 1) % di2
                    copy(auto_copy(), txSy[:, :, i_tile_next, smem_pipe_read], txry1[:, :, (i + 1) % 2])
                    copy(auto_copy(), txSw2[:, :, i_tile_next, smem_pipe_read], txrw2[:, :, (i + 1) % 2])
                    if i == 0:
                        if j + stages - 1 < do2:
                            copy(auto_copy((bl, bn)), txgw2[:, :, j + stages - 1], txsw2[:, :, smem_pipe_write])
                            acquire_seq_semaphore(wait_counter + j + stages - 1, 1)
                            copy(auto_copy((bm, bn)), txgy1[:, :, j + stages - 1], txsy[:, :, smem_pipe_write])
                        smem_pipe_write = smem_pipe_read
                        cp_async_commit_group()

                    if i == di2 - 2:
                        smem_pipe_read += 1
                        smem_pipe_read = 0 if smem_pipe_read == stages else smem_pipe_read

                    mma(tiled_mma_yw2, tr_yw2, txry1[:, :, i % 2], txrw2[:, :, i % 2], tr_yw2)
            tr_z = cast(tr_yw2, f16)
            tg_z = tensor_view(z[:, pid_l * bl :], TensorLayout((bm, bl), (dmodel, 1)), "global")
            cute_atomic_add(tr_z, tg_z)

    func = script_module.build()
    # a_mem = hidet.empty([m, k], device="cuda")
    # b_mem = hidet.empty([k, n], device="cuda")
    # c_mem = hidet.empty([m, n], device="cuda")
    # func(a_mem, b_mem, c_mem)
    return func


def data(batch_size, seqlen, dmodel, dffn, dtype="float16", device="cuda", return_hidet=False):
    dtype = getattr(torch, dtype)
    lo = -3
    hi = 3
    x = torch.randint(low=lo, high=hi, size=(batch_size * seqlen, dmodel), dtype=dtype, device=device)
    w1 = torch.randint(low=lo, high=hi, size=(dmodel, 2 * dffn), dtype=dtype, device=device)
    w2 = torch.randint(low=lo, high=hi, size=(dffn, dmodel), dtype=dtype, device=device)
    z = torch.empty((batch_size * seqlen, dmodel), dtype=dtype, device=device)
    y = torch.empty((batch_size * seqlen, dffn), dtype=dtype, device=device)

    w1 = w1 / dmodel
    if return_hidet:
        x = hidet.from_torch(x)
        w1 = hidet.from_torch(w1)
        w2 = hidet.from_torch(w2)
        z = hidet.from_torch(z)
        y = hidet.from_torch(y)

    return x, w1, w2, z, y


@pytest.mark.parametrize("batch_size, seqlen, dmodel, dffn", [(8, 1, 4096, 4096 * 4), (8, 1, 1024, 1024 * 4)])
def test_mlp(batch_size, seqlen, dmodel, dffn):
    mlp(batch_size, seqlen, dmodel, dffn)


@pytest.mark.parametrize("batch_size, seqlen, dmodel, dffn", [(8, 1, 4096, 4096 * 4), (8, 1, 1024, 1024 * 4)])
def test_mlp_v2(batch_size, seqlen, dmodel, dffn):
    mlp_v2(batch_size, seqlen, dmodel, dffn)


# this kernel requires inter threadblock communication, so we should limit the
# block size. Otherwise, deadlock may occur.
@pytest.mark.parametrize("batch_size, seqlen, dmodel, dffn", [(8, 1, 4096, 4096 * 4), (8, 1, 4096, 4096 * 3)])
def test_mlp_v3(batch_size, seqlen, dmodel, dffn):
    # hidet.option.cache_dir("./demo_mlp")
    # hidet.option.search_space(2)
    # hidet.option.debug_cache_tuning()
    # hidet.option.save_lower_ir(True)
    func = mlp_v3(batch_size, seqlen, dmodel, dffn)
    x, w1, w2, z, y = data(batch_size, seqlen, dmodel, dffn)
    bn = 128
    blocks = (dffn + bn - 1) // bn
    lock = torch.empty((blocks,), dtype=torch.int32, device="cuda")
    mean, min_lat, max_lat = bench(func, (x, w1, w2, z, y, lock))
    from hidet.ir.dtypes import f16

    memory = f16.nbytes * (batch_size * seqlen * dmodel * 2 + dmodel * dffn * 3)
    print("time={:.3f} ms, bandwidth={:.3f} GB/s".format(mean, memory / (1e6 * mean)))

    def fn():
        z.zero_()
        func(x, w1, w2, z, y, lock)

    from hidet.utils.benchmark import do_bench

    mean = do_bench(fn, percentiles=None)
    print("time={:.3f} ms, bandwidth={:.3f} GB/s".format(mean, memory / (1e6 * mean)))
    fn()

    from torch.nn import SiLU

    silu_ = SiLU()

    def fn():
        y = x @ w1
        y1 = y[:, :dffn]
        y2 = y[:, dffn:]
        y3 = silu_(y1) * y2
        return y3 @ w2

    from hidet.utils.benchmark import do_bench

    mean, min_lat, max_lat = bench(fn, ())
    #    mean = do_bench(fn, percentiles=None)
    print("time={:.3f} ms, bandwidth={:.3f} GB/s".format(mean, memory / (1e6 * mean)))

    z2 = fn()
    import numpy as np

    np.set_printoptions(threshold=3000, linewidth=200, edgeitems=100)
    np.testing.assert_allclose(actual=z.cpu().numpy(), desired=z2.cpu().numpy(), rtol=1e-2)
