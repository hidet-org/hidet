from typing import Union, List, Tuple

import hidet
from hidet.ir.type import DataType, TensorType, tensor_type, tensor_pointer_type
from hidet.ir.dtypes import vectorize
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
    from hidet.lang.cuda import dynamic_shared_memory

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
        .loc(1, block_warps[0], 1)
        .col(1, warp_repeat[0], 1)
        .loc(1, warp_threads[0], 1)
        .row(1, thread_shape[0], 1)
    )
    regs_b_layout = ( # 2 x 1 x block_n
        row_major(2, 1, 1)
        .loc(1, 1, block_warps[1])
        .row(1, 1, warp_repeat[1])
        .loc(1, 1, warp_threads[1])
        .row(1, 1, thread_shape[1])
    )
    regs_c_layout = ( # block_m x block_n
        local_layout(block_warps[0], block_warps[1])
        .row(warp_repeat[0], warp_repeat[1])
        .loc(warp_threads[0], warp_threads[1])
        .row(thread_shape[0], thread_shape[1])
    )
    tune.check(num_threads % block_k == 0)
    tune.check(block_m % (num_threads // block_k) == 0)
    tune.check(block_n % (num_threads // block_k) == 0)
    regs_a_ldg_layout = local_layout(num_threads // block_k, block_k).row_major(block_m // (num_threads // block_k), 1)
    regs_b_ldg_layout = row_major(1, block_n // (num_threads // block_k)).local(block_k, num_threads // block_k)

    smem_total_size: Expr = dtype.nbytes * (smem_a_layout.size + smem_b_layout.size)

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

        # copy the first k-tile from global to shared memory





