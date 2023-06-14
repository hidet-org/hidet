from typing import Union, List, Tuple

import hidet
from hidet.ir.type import DataType, TensorType, tensor_type, tensor_pointer_type
from hidet.ir.dtypes import vectorize, i32
from hidet.ir.expr import Expr, is_true, tensor_pointer_var
from hidet.ir.layout import row_major, data_layout, local_layout
from hidet.ir.library import tune
from hidet.ir.library.utils import get_tensor_type
from hidet.ir.utils.broadcast_utils import can_broadcast
from hidet.utils import prod


def check_type(a_type: TensorType, b_type: TensorType, c_type: TensorType):
    if not (a_type.dtype == b_type.dtype == c_type.dtype):
        raise TypeError(
            'The data type of a, b, c should be the same, got {}, {}, {}'.format(
                a_type.dtype, b_type.dtype, c_type.dtype
            )
        )
    a_shape = a_type.shape
    b_shape = b_type.shape
    c_shape = c_type.shape
    if len(a_shape) < 2 or len(b_shape) < 2 or len(c_shape) < 2:
        raise ValueError(
            'The rank of a, b, c should be greater than 1, got {}, {}, {}'.format(
                len(a_shape), len(b_shape), len(c_shape)
            )
        )

    if (
        is_true(a_shape[-1] != b_shape[-2])
        or is_true(a_shape[-2] != c_shape[-2])
        or is_true(b_shape[-1] != c_shape[-1])
        or not can_broadcast(a_shape[:-2], c_shape[:-2])
        or not can_broadcast(b_shape[:-1], c_shape[:-1])
    ):
        raise ValueError('The shapes of a, b, c are not compatible, got {}, {}, {}'.format(a_shape, b_shape, c_shape))


def matmul_simt(
    arg_a: Expr,
    arg_b: Expr,
    arg_c: Expr,
    *,
    parallel_k=1,
    block_shape=(128, 128, 16),
    warp_shape=(64, 64, 8),
    warp_threads=(4, 8),
    thread_shape=(4, 4),
    arch='sm_70'
):
    from hidet.lang import attrs
    from hidet.lang import tensor_pointer, register_tensor, grid, cast, deref, meta
    from hidet.lang.cuda import dynamic_shared_memory, threadIdx, blockIdx, syncthreads
    from hidet.lang.mapping import spatial, repeat

    capability = hidet.cuda.capability(arch)
    a_type: TensorType = get_tensor_type(arg_a)
    b_type: TensorType = get_tensor_type(arg_b)
    c_type: TensorType = get_tensor_type(arg_c)
    check_type(a_type, b_type, c_type)
    dtype: DataType = a_type.dtype

    if dtype.nbytes < 4:
        lanes: int = 4 // dtype.nbytes
        vtype: DataType = vectorize(dtype, lanes)
    else:
        lanes: int = 1
        vtype: DataType = dtype

    a_head: List[Expr] = list(a_type.shape[:-2])
    b_head: List[Expr] = list(b_type.shape[:-2])
    c_head: List[Expr] = list(c_type.shape[:-2])
    m_size: Expr = a_type.shape[-2]
    n_size: Expr = b_type.shape[-1]
    k_size: Expr = a_type.shape[-1]

    tune.check(all(a % b == 0 for a, b in zip(block_shape, warp_shape)))
    tune.check(all(a % (b * c) == 0 for a, b, c in zip(warp_shape[:2], warp_threads, thread_shape[:2])))
    tune.check(not is_true(k_size % lanes != 0))
    tune.check(not is_true(n_size % lanes != 0))
    block_warps = tuple(a // b for a, b in zip(block_shape, warp_shape))  # Tuple[int, int, int]
    warp_repeat = tuple(a // (b * c) for a, b, c in zip(warp_shape[:2], warp_threads, thread_shape[:2]))  # Tuple[int, int]

    num_warps: int = prod(block_warps)
    num_threads: int = num_warps * 32
    block_m, block_n, block_k = block_shape
    warp_m, warp_n, warp_k = warp_shape
    block_warps_m, block_warps_n, block_warps_k = block_m // warp_m, block_n // warp_n, block_k // warp_k
    block_tiles_m: Expr = (m_size + block_m - 1) // block_m
    block_tiles_n: Expr = (n_size + block_n - 1) // block_n
    tune.check(num_threads <= capability.maxThreadsPerBlock)

    # prepare data layout
    tune.check(block_k % lanes == 0)
    block_k_vectors = block_k // lanes
    smem_a_layout = data_layout([2, block_k_vectors, block_m])
    smem_b_layout = data_layout([2, block_k_vectors, block_n])
    regs_a_layout = (  # 2 x block_m
        row_major(2, 1)
        .local(1, block_warps[0])
        .column_major(1, warp_repeat[0])
        .local(1, warp_threads[0])
        .row_major(1, thread_shape[0])
    )
    regs_b_layout = (  # 2 x block_n
        row_major(2, 1)
        .local(1, block_warps[1])
        .row_major(1, warp_repeat[1])
        .local(1, warp_threads[1])
        .row_major(1, thread_shape[1])
    )
    regs_c_layout = (  # block_m x block_n
        local_layout(block_warps[0], block_warps[1])
        .row_major(warp_repeat[0], warp_repeat[1])
        .local(warp_threads[0], warp_threads[1])
        .row_major(thread_shape[0], thread_shape[1])
    )
    # regs_a_ldg: (block_k / lanes) x block_m (for vectorized dtype)
    if num_threads >= block_k_vectors:
        tune.check(num_threads % block_k_vectors == 0)
        rows = num_threads // block_k_vectors
        tune.check(block_m % rows == 0)
        regs_a_ldg_layout = row_major(1, block_m // rows).local(block_k_vectors, rows)
        a_g2s_mapping = repeat(1, block_m // rows).spatial(block_k_vectors, rows, ranks=[1, 0])
    else:
        regs_a_ldg_layout = None
        tune.check(False)   # rare case

    # regs_b_ldg: block_k x block_n
    if num_threads >= block_n:
        tune.check(num_threads % block_n == 0)
        rows = num_threads // block_n
        tune.check(block_k % rows == 0)
        regs_b_ldg_layout = row_major(block_k // rows, 1).local(rows, block_n)
        b_g2s_mapping = repeat(block_k // rows, 1).spatial(rows, block_n)
    else:
        tune.check(block_n % num_threads == 0)
        segments = block_n // num_threads
        tune.check(block_n % segments == 0)
        regs_b_ldg_layout = row_major(block_k, segments).local(1, num_threads)
        b_g2s_mapping = repeat(block_k, segments).spatial(1, num_threads)

    # prepare task mapping
    # block_m x block_n
    block_mapping = (
        spatial(block_warps[0], block_warps[1])
        .repeat(warp_repeat[0], warp_repeat[1])
        .spatial(warp_threads[0], warp_threads[1])
        .spatial(thread_shape[0], thread_shape[1])
    )
    # block_m x *
    a_s2r_mapping = (
        spatial(*block_warps[:2]).repeat(warp_repeat[0], 1).spatial(*warp_threads).repeat(thread_shape[0], 1)
    )
    # * x block_n
    b_s2r_mapping = (
        spatial(*block_warps[:2]).repeat(1, warp_repeat[1]).spatial(*warp_threads).repeat(1, thread_shape[1])
    )

    smem_total_size: Expr = dtype.nbytes * (smem_a_layout.size + smem_b_layout.size)

    @hidet.script
    def copy_a_g2r(  # copy tile block_m x block_k of matrix a from global memory to shared memory
        a: tensor_type(dtype, shape=a_head + [m_size, k_size]),
        regs_a_ldg: tensor_type(vtype, layout=regs_a_ldg_layout),
        offset_m: i32,
        offset_k: i32,
        first_k_tile_size: i32,
    ):
        gmem_a = a[blockIdx.y, offset_m:, offset_k:]
        for kv, i in a_g2s_mapping.on(threadIdx.x):
            k_predicate = (first_k_tile_size == 0) or kv < first_k_tile_size
            if offset_m + i < m_size and k_predicate:
                regs_a_ldg[kv, i] = deref(cast(~gmem_a[i, kv * lanes], ~vtype))  # vectorized load
            else:
                regs_a_ldg[kv, i] = vtype.zero

    @hidet.script
    def copy_a_r2s(
        regs_a_ldg: tensor_type(dtype=vtype, layout=regs_a_ldg_layout),
        smem_a: tensor_type(dtype=vtype, layout=smem_a_layout),
        buffer_idx: i32,
    ):
        for i, k in a_g2s_mapping.on(threadIdx.x):
            smem_a[buffer_idx, i, k] = regs_a_ldg[i, k]

    @hidet.script
    def copy_a_s2r(
        smem_a: tensor_type(dtype=vtype, layout=smem_b_layout),
        regs_a: tensor_type(dtype=vtype, layout=regs_a_layout),
        smem_buffer_idx: i32,
        regs_buffer_idx: i32,
        k_frag_idx: i32,
    ):
        _, _, warp_idx_k = spatial(*block_warps).map(threadIdx.x // 32)
        smem_a_slice = smem_a[smem_buffer_idx, warp_idx_k * (warp_k // lanes):, :]
        for i, _ in a_s2r_mapping.on(threadIdx.x):
            regs_a[regs_buffer_idx, i] = smem_a_slice[k_frag_idx, i]

    @hidet.script
    def copy_b_g2r(
        b: tensor_type(dtype=dtype, shape=b_head + [k_size, n_size]),
        regs_b_ldg: tensor_type(dtype=dtype, layout=regs_b_ldg_layout),
        offset_k: i32,
        offset_n: i32,
        first_k_tile_size: i32,
    ):
        gmem_b = b[blockIdx.y, offset_k:, offset_n:]
        for k, j in b_g2s_mapping.on(threadIdx.x):
            k_predicate = (first_k_tile_size == 0) or k < first_k_tile_size
            if offset_n + j < n_size and k_predicate:
                regs_b_ldg[k, j] = gmem_b[k, j]
            else:
                regs_b_ldg[k, j] = dtype.zero

    @hidet.script
    def copy_b_r2s(
        regs_b_ldg: tensor_type(dtype=dtype, layout=regs_b_ldg_layout),
        smem_b: tensor_type(dtype=vtype, layout=smem_b_layout),
        buffer_idx: i32,
    ):
        for k, j in b_g2s_mapping.on(threadIdx.x):
            dst_ptr = cast(~smem_b[buffer_idx, k // lanes, j], ~dtype)
            dst_ptr[k % lanes] = regs_b_ldg[k, j]

    @hidet.script
    def copy_b_s2r(
        smem_b: tensor_type(dtype=vtype, layout=smem_b_layout),
        regs_b: tensor_type(dtype=vtype, layout=regs_b_layout),
        smem_buffer_idx: i32,
        regs_buffer_idx: i32,
        k_frag_idx: i32,
    ):
        _, _, warp_idx_k = spatial(*block_warps).map(threadIdx.x // 32)
        smem_b_slice = smem_b[smem_buffer_idx, warp_idx_k * (warp_k // lanes):, :]
        for _, j in b_s2r_mapping.on(threadIdx.x):
            regs_b[regs_buffer_idx, j] = smem_b_slice[k_frag_idx, j]

    @hidet.script
    def copy_c_r2g(
        regs_c: tensor_type(dtype=vtype, layout=regs_c_layout),
        c: tensor_type(dtype=dtype, shape=c_head + [m_size, n_size]),
        offset_m: i32,
        offset_n: i32,
    ):
        gmem_c = c[blockIdx.y, offset_m:, offset_n:]
        for i, j in block_mapping.on(threadIdx.x):
            if offset_m + i < m_size and offset_n + j < n_size:
                regs_c_ptr = cast(~regs_c[i, j], ~dtype)
                result = dtype.zero
                for k in range(lanes):
                    result += regs_c_ptr[k]
                gmem_c[i, j] = result

    @hidet.script
    def mma(
        regs_a: tensor_type(dtype=vtype, layout=regs_a_layout),
        regs_b: tensor_type(dtype=vtype, layout=regs_b_layout),
        regs_c: tensor_type(dtype=vtype, layout=regs_c_layout),
        buffer_idx: i32,
    ):
        for i, j in block_mapping.on(threadIdx.x):
            regs_c[i, j] += regs_a[buffer_idx, i] * regs_b[buffer_idx, j]

    @hidet.script
    def matmul_kernel(
        a: dtype[a_head + [m_size, k_size]], b: dtype[b_head + [k_size, n_size]], c: dtype[c_head + [m_size, n_size]]
    ):
        attrs.func_kind = 'public'
        attrs.cuda.block_dim = num_threads
        attrs.cuda.grid_dim = block_tiles_m * block_tiles_n, prod(c_head)
        attrs.cuda.dynamic_smem_bytes = smem_total_size

        smem = tensor_pointer('uint8', [smem_total_size], init=dynamic_shared_memory(0, 'uint8'))
        smem_a = tensor_pointer(vtype, layout=smem_a_layout, init=~smem[0])
        smem_b = tensor_pointer(vtype, layout=smem_b_layout, init=~smem[vtype.nbytes * smem_a_layout.size])
        regs_a = register_tensor(vtype, layout=regs_a_layout)
        regs_b = register_tensor(vtype, layout=regs_b_layout)
        regs_c = register_tensor(vtype, layout=regs_c_layout)
        regs_a_ldg = register_tensor(vtype, layout=regs_a_ldg_layout)
        regs_b_ldg = register_tensor(dtype, layout=regs_b_ldg_layout)

        offset_m, offset_n = spatial(block_tiles_m, block_tiles_n).map(blockIdx.x)
        k_tiles = (k_size + block_k - 1) / block_k

        # Copy first k-tile from global to shared
        first_k_tile_size = k_size - (k_tiles - 1) * block_k
        copy_a_g2r(a, regs_a_ldg, offset_m, 0, first_k_tile_size)
        copy_a_r2s(regs_a_ldg, smem_a, 0)
        copy_b_g2r(b, regs_b_ldg, 0, offset_n, first_k_tile_size)
        copy_b_r2s(regs_b_ldg, smem_b, 0)
        syncthreads()
        # Copy first k-frag within first k-tile from shared to local
        copy_a_s2r(smem_a, regs_a, 0, 0, 0)
        copy_b_s2r(smem_b, regs_b, 0, 0, 0)
        syncthreads()
        # Initialize regs C
        for i, j in block_mapping.on(threadIdx.x):
            regs_c[i, j] = 0.0

        # Main k loop
        for k0 in range(k_tiles - 1):  # iterate block_k tile
            offset_k = k0 * block_k + first_k_tile_size
            warp_k_tiles = warp_k // lanes
            for k1 in grid(warp_k_tiles):
                if k1 == warp_k_tiles - 1:
                    # Store next AB tile from local into shared
                    copy_a_r2s(regs_a_ldg, smem_a, (k0 + 1) % 2)
                    copy_b_r2s(regs_b_ldg, smem_b, (k0 + 1) % 2)
                    syncthreads()
                    # Load next k-fragment (from next k-tile) from shared to local
                    copy_a_s2r(smem_a, regs_a, (k0 + 1) % 2, (k1 + 1) % 2, 0)
                    copy_b_s2r(smem_b, regs_b, (k0 + 1) % 2, (k1 + 1) % 2, 0)
                else:
                    # Load next k-fragment from shared to local
                    copy_a_s2r(smem_a, regs_a, k0 % 2, (k1 + 1) % 2, k1 + 1)
                    copy_b_s2r(smem_b, regs_b, k0 % 2, (k1 + 1) % 2, k1 + 1)
                if k1 == 0:
                    # Load next AB tile from global into local
                    copy_a_g2r(a, regs_a_ldg, offset_m, offset_k, 0)
                    copy_b_g2r(b, regs_b_ldg, offset_k, offset_n, 0)
                # Perform MMA
                mma(regs_a, regs_b, regs_c, k1 % 2)
        # Perform MMA for last k-tile
        last_k = k_tiles - 1
        for k1 in grid(block_warps_k):
            if k1 < block_warps_k - 1:
                copy_a_s2r(smem_a, regs_a, last_k % 2, (k1 + 1) % 2, k1 + 1)
                copy_b_s2r(smem_b, regs_b, last_k % 2, (k1 + 1) % 2, k1 + 1)
            mma(regs_a, regs_b, regs_c, k1 % 2)

        # Store results from regs_c into C
        copy_c_r2g(regs_c, c, offset_m, offset_n)

    from hidet.lang.script import ScriptModuleContext
    return ScriptModuleContext.current_context().lookup('matmul_kernel')(arg_a, arg_b, arg_c)
