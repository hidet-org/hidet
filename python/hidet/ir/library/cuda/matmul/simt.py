from typing import Union, List, Tuple

import hidet
from hidet.ir.type import DataType, TensorType, tensor_type, tensor_pointer_type
from hidet.ir.dtypes import vectorize, i32
from hidet.ir.expr import Expr, is_true, tensor_pointer_var
from hidet.ir.layout import row_major, data_layout, local_layout
from hidet.graph.ops.utils import tune
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
        or is_true(a_shape[:-2] == c_shape[:-2])
        or is_true(b_shape[:-1] == c_shape[:-1])
        or not can_broadcast(a_shape[:-2], c_shape[:-2])
        or not can_broadcast(b_shape[:-1], c_shape[:-1])
    ):
        raise ValueError('The shape of a, b, c should be compatible, got {}, {}, {}'.format(a_shape, b_shape, c_shape))


def matmul_simt(
    arg_a: Expr,
    arg_b: Expr,
    arg_c: Expr,
    *,
    parallel_k=1,
    block_shape=(128, 128, 16),
    warp_shape=(64, 64, 16),
    warp_threads=(4, 8),
    thread_shape=(4, 4),
    arch='sm_70'
):
    from hidet.lang import attrs
    from hidet.lang import tensor_pointer, register_tensor
    from hidet.lang.cuda import dynamic_shared_memory, threadIdx, blockIdx
    from hidet.lang.mapping import spatial, repeat


    capability = hidet.cuda.capability(arch)
    a_type: TensorType = get_tensor_type(arg_a)
    b_type: TensorType = get_tensor_type(arg_b)
    c_type: TensorType = get_tensor_type(arg_c)
    check_type(a_type, b_type, c_type)
    dtype: DataType = a_type.dtype

    if dtype.nbytes < 4:
        lanes: int = 4 // dtype.nbytes
        vec_dtype: DataType = vectorize(dtype, lanes)
    else:
        lanes: int = 1
        vec_dtype: DataType = dtype

    a_head: List[Expr] = list(a_type.shape[:-2])
    b_head: List[Expr] = list(b_type.shape[:-2])
    c_head: List[Expr] = list(c_type.shape[:-2])
    m_size: Expr = a_type.shape[-2]
    n_size: Expr = b_type.shape[-1]
    k_size: Expr = a_type.shape[-1]

    tune.check(all(a % b == 0 for a, b in zip(block_shape, warp_shape)))
    tune.check(all(a % (b * c) == 0 for a, b, c in zip(warp_shape[:2], warp_threads, thread_shape)))
    tune.check(is_true(k_size % lanes == 0))
    tune.check(is_true(n_size % lanes == 0))
    block_warps = tuple(a // b for a, b in zip(block_shape, warp_shape))  # Tuple[int, int, int]
    warp_repeat = tuple(a // (b * c) for a, b, c in zip(warp_shape[:2], warp_threads, thread_shape))  # Tuple[int, int]

    num_warps: int = prod(block_warps)
    num_threads: int = num_warps * 32
    block_m, block_n, block_k = block_warps
    warp_m, warp_n, warp_k = warp_shape
    block_tiles_m: Expr = (m_size + block_shape[0] - 1) // block_shape[0]
    block_tiles_n: Expr = (n_size + block_shape[1] - 1) // block_shape[1]
    tune.check(num_threads <= capability.maxThreadsPerBlock)

    # prepare data layout
    smem_a_layout = data_layout([2, block_m, block_k], ranks=[0, 2, 1])
    smem_b_layout = data_layout([2, block_k, block_n], ranks=[0, 1, 2])
    regs_a_layout = ( # 2 x block_m x 1
        row_major(2, 1, 1)
        .local(1, block_warps[0], 1)
        .column_major(1, warp_repeat[0], 1)
        .local(1, warp_threads[0], 1)
        .row_major(1, thread_shape[0], 1)
    )
    regs_b_layout = ( # 2 x 1 x block_n
        row_major(2, 1, 1)
        .local(1, 1, block_warps[1])
        .row_major(1, 1, warp_repeat[1])
        .local(1, 1, warp_threads[1])
        .row_major(1, 1, thread_shape[1])
    )
    regs_c_layout = ( # block_m x block_n
        local_layout(block_warps[0], block_warps[1])
        .row_major(warp_repeat[0], warp_repeat[1])
        .local(warp_threads[0], warp_threads[1])
        .row_major(thread_shape[0], thread_shape[1])
    )
    tune.check(num_threads % block_k == 0)
    tune.check(block_m % (num_threads // block_k) == 0)
    tune.check(block_n % (num_threads // block_k) == 0)
    lines = num_threads // block_k
    regs_a_ldg_layout = local_layout(lines, block_k).row_major(block_m // lines, 1)
    regs_b_ldg_layout = row_major(1, block_n // lines).local(block_k, lines)

    # prepare task mapping
    block_mapping = (
        spatial(block_warps[0], block_warps[1])
        .repeat(warp_repeat[0], warp_repeat[1])
        .spatial(warp_threads[0], warp_threads[1])
        .spatial(thread_shape[0], thread_shape[1])
    )
    a_g2s_mapping = spatial(lines, block_k) * repeat(block_shape[0] // lines, 1)
    b_g2s_mapping = repeat(1, block_shape[1] // lines) * spatial(block_k, lines)
    a_s2r_mapping = (
        spatial(*block_warps).repeat(warp_repeat[0], 1).spatial(*warp_threads).repeat(thread_shape[0], 1)
    )
    b_s2r_mapping = (
        spatial(*block_warps).repeat(1, warp_repeat[1]).spatial(*warp_threads).repeat(1, thread_shape[1])
    )

    smem_total_size: Expr = dtype.nbytes * (smem_a_layout.size + smem_b_layout.size)

    @hidet.script
    def copy_a_g2r(
        a: tensor_type(dtype, shape=a_head + [m_size, k_size]),
        regs_a_ldg: tensor_type(dtype, layout=regs_a_ldg_layout),
        offset_m: i32,
        offset_k: i32,
        first_k_tile_size: i32,
    ):
        gmem_a = a[blockIdx.y, offset_m:, offset_k:]
        for i, k in a_g2s_mapping.on(threadIdx.x):
            k_predicate = (first_k_tile_size == 0) or k < first_k_tile_size
            if offset_m + i < m_size and k_predicate:
                regs_a_ldg[i, k] = gmem_a.read([i, k], protected=False)
            else:
                regs_a_ldg[i, k] = 0.0

    @hidet.script
    def copy_a_r2s(
        regs_a_ldg: tensor_type(dtype=dtype, layout=regs_a_ldg_layout),
        smem_a: tensor_type(dtype=dtype, layout=data_layout([2, block_m, block_k], ranks=[0, 2, 1])),
        buffer_idx: i32,
    ):
        for i, k in a_g2s_mapping.on(threadIdx.x):
            smem_a[buffer_idx, i, k] = regs_a_ldg[i, k]

    @hidet.script
    def copy_a_s2r(
        smem_a: tensor_type(dtype=dtype, layout=data_layout([2, block_m, block_k], ranks=[0, 2, 1])),
        regs_a: tensor_type(dtype=dtype, layout=regs_a_layout),
        smem_buffer_idx: i32,
        regs_buffer_idx: i32,
        k_frag_idx: i32,
    ):
        smem_a_start = smem_a[smem_buffer_idx, :, k_frag_idx:]
        for i, k in a_s2r_mapping.on(threadIdx.x):
            regs_a[regs_buffer_idx, i, k] = smem_a_start[i, 0]

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
                regs_b_ldg[k, j] = gmem_b.read([k, j], protected=False)
            else:
                regs_b_ldg[k, j] = 0.0

    @hidet.script
    def copy_b_r2s(
        regs_b_ldg: tensor_type(dtype=dtype, layout=regs_b_ldg_layout),
        smem_b: tensor_type(dtype=dtype, layout=data_layout([2, block_k, block_n], ranks=[0, 1, 2])),
        buffer_idx: i32,
    ):
        for k, j in b_g2s_mapping.on(threadIdx.x):
            smem_b[buffer_idx, k, j] = regs_b_ldg[k, j]

    @hidet.script
    def copy_b_s2r(
        smem_b: tensor_type(dtype=dtype, layout=data_layout([2, block_k, block_n], ranks=[0, 1, 2])),
        regs_b: tensor_type(dtype=dtype, layout=regs_b_layout),
        smem_buffer_idx: i32,
        regs_buffer_idx: i32,
        k_frag_idx: i32,
    ):
        smem_b_start = smem_b[smem_buffer_idx, k_frag_idx:, :]
        for k, j in b_s2r_mapping.on(threadIdx.x):
            regs_b[regs_buffer_idx, k, j] = smem_b_start[0, j]

    @hidet.script
    def copy_c_r2g(
        regs_c: tensor_type(dtype=dtype, layout=regs_c_layout),
        c: tensor_type(dtype=dtype, shape=c_head + [m_size, n_size]),
        offset_m: i32,
        offset_n: i32,
    ):
        gmem_c = c[blockIdx.y, offset_m:, offset_n:]
        for i, j in block_mapping.on(threadIdx.x):
            if offset_m + i < m_size and offset_n + j < n_size:
                gmem_c.write([i, j], regs_c[i, j], protected=False)

    @hidet.script
    def mma(
        regs_a: tensor_type(dtype=dtype, layout=regs_a_layout),
        regs_b: tensor_type(dtype=dtype, layout=regs_b_layout),
        regs_c: tensor_type(dtype=dtype, layout=regs_c_layout),
        buffer_idx: i32,
    ):
        for i, j in block_mapping.on(threadIdx.x):
            for k in range(warp_k):
                regs_c[i, j] += regs_a[buffer_idx, i, k] * regs_b[buffer_idx, k, j]

    @hidet.script
    def matmul_kernel(
        a: dtype[a_head + [m_size, k_size]], b: dtype[b_head + [k_size, n_size]], c: dtype[c_head + [m_size, n_size]]
    ):
        attrs.func_kind = 'public'
        attrs.cuda.block_dim = num_threads
        attrs.cuda.grid_dim = block_tiles_m * block_tiles_n, prod(c_head)
        attrs.cuda.dynamic_smem_bytes = smem_total_size

        smem = tensor_pointer('uint8', [smem_total_size], init=dynamic_shared_memory(0, 'uint8'))
        smem_a = tensor_pointer(dtype, layout=smem_a_layout, init=~smem[0])
        smem_b = tensor_pointer(dtype, layout=smem_b_layout, init=~smem[dtype.nbytes * smem_a_layout.size])
        regs_a = register_tensor(dtype, layout=regs_a_layout)
        regs_b = register_tensor(dtype, layout=regs_b_layout)
        regs_c = register_tensor(dtype, layout=regs_c_layout)
        regs_a_ldg = register_tensor(dtype, layout=regs_a_ldg_layout)
        regs_b_ldg = register_tensor(dtype, layout=regs_b_ldg_layout)

        # initialize regs c
        for i, j in block_mapping.on(threadIdx.x):
            regs_c[i, j] = dtype(0)

        # copy the first k-tile from global to shared memory
        k_tiles = (k_size + block_k - 1) / block_k
        first_k_tile_size = k_size - (k_tiles - 1) * block_k



