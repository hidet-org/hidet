# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Dict, List, Tuple, Union
from functools import partial

import hidet
from hidet.ir import dtypes
from hidet.ir.type import DataType, data_type
from hidet.ir.dtypes import float16, float32, bfloat16, i32
from hidet.ir.expr import if_then_else, Int, Expr
from hidet.ir.module import IRModule
from hidet.ir.compute import TensorNode
from hidet.ir.task import Task
from hidet.ir.compute import compute, reduce
from hidet.graph.ops.utils import input_like, broadcast_shape, can_mutually_broadcast
from hidet.ir.library import tune
from hidet.graph.operator import Operator, Tensor
from hidet.utils.py import cdiv, prod
from hidet.graph.ops.utils import broadcast_indices
from hidet.utils import initialize

from hidet.ir.cute.ops import (
    make_tensor,
    tensor_view,
    partition_src,
    partition_dst,
    mask,
    copy,
    mma,
    rearrange,
    fill,
    cast,
)
from hidet.ir.cute.collective import collective_store
from hidet.ir.cute import auto_layout, layout_auto, auto_copy
from hidet.ir.cute.layout import Level, TensorLayout, ThrValAtom, TiledTensorLayout
from hidet.ir.cute.algorithm import MmaAtom, TiledMma, TiledCopy

from hidet.lang import attrs, grid
from hidet.lang.cuda import blockIdx, syncthreads
from hidet.lang.cuda import cp_async_wait_all, cp_async_commit_group, cp_async_wait_group
from hidet.lang.constructs.declare import as_tensor_pointer
from hidet.lang.mapping import spatial
from hidet.utils.py import is_power_of_two


_tiled_mma_space_1: List[TiledMma] = []
_tiled_mma_space_2: List[TiledMma] = []


@initialize()
def register_tiled_mma():
    a = TensorLayout(((4, 8), (2, 2, 2)), ((32, 1), (16, 8, 128)))
    b = TensorLayout(((4, 8), (2, 2)), ((16, 1), (8, 64)))
    c = TensorLayout(((4, 8), (2, 2)), ((32, 1), (16, 8)))
    mma_atom = MmaAtom("warp", (16, 8, 16), a, b, c, c, (1, 2))
    warp_in_threadblock = Level("warp", "thread_block", (2, 2), TensorLayout((2, 2)), (4, 4))
    tiled_mma = TiledMma(mma_atom, [warp_in_threadblock])
    _tiled_mma_space_1.append(tiled_mma)

    def get_warp(tile: int):
        if tile % 8 == 0:
            return 8
        elif tile % 4 == 0:
            return 4
        elif tile % 2 == 0:
            return 2
        else:
            return 1

    for tile_n in [1, 2, 3, 4, 6, 8, 12, 16]:
        a = TensorLayout(((4, 8), (2, 2)), ((16, 1), (8, 64)))
        b = TensorLayout(((4, 8), (2, 2, 2)), ((32, 1), (16, 8, 128)))
        c = TensorLayout(((4, 8), (2, 2)), ((2, 8), (1, 64)))
        mma_atom = MmaAtom("warp", (8, 16, 16), a, b, c, c, (1, 1))
        warp_n = get_warp(tile_n)
        if (tile_n // warp_n) % 2 != 0:
            continue
        warp_in_threadblock = Level(
            "warp", "thread_block", (1, warp_n), TensorLayout((1, warp_n)), (1, tile_n // warp_n)
        )
        tiled_mma = TiledMma(mma_atom, [warp_in_threadblock])
        if is_power_of_two(tile_n):
            _tiled_mma_space_1.append(tiled_mma)
        else:
            _tiled_mma_space_2.append(tiled_mma)

    for tile_m in [1, 2, 3, 4, 6, 8, 12, 16]:
        a = TensorLayout(((4, 8), (2, 2, 2)), ((32, 1), (16, 8, 128)))
        b = TensorLayout(((4, 8), (2, 2)), ((16, 1), (8, 64)))
        c = TensorLayout(((4, 8), (2, 2)), ((32, 1), (16, 8)))
        mma_atom = MmaAtom("warp", (16, 8, 16), a, b, c, c, (1, 1))
        warp_m = get_warp(tile_m)
        if (tile_m // warp_m) % 2 != 0:
            continue
        warp_in_threadblock = Level(
            "warp", "thread_block", (warp_m, 1), TensorLayout((warp_m, 1)), (tile_m // warp_m, 1)
        )
        tiled_mma = TiledMma(mma_atom, [warp_in_threadblock])
        if is_power_of_two(tile_m):
            _tiled_mma_space_1.append(tiled_mma)
        else:
            _tiled_mma_space_2.append(tiled_mma)

    compute_bound = [
        (256, 128),
        (128, 256),
        (256, 64),
        (64, 256),
        (128, 128),
        (64, 128),
        (128, 64),
        (128, 32),
        (64, 32),
    ]

    for warp_m in [1, 2, 4]:
        for warp_n in [1, 2, 4]:
            for repeat_m in [1, 2, 3, 4, 6, 8, 12, 16]:
                for repeat_n in [1, 2, 3, 4, 6, 8, 12, 16]:
                    if warp_m == 2 and warp_n == 2 and repeat_m == 4 and repeat_n == 4:
                        continue
                    if warp_m * warp_n > 8:
                        continue
                    num_regs = repeat_m * repeat_n * 8 + 2 * repeat_m * 4 + 2 * repeat_n * 2
                    if repeat_m * repeat_n < 2 or num_regs > 255:
                        continue
                    a = TensorLayout(((4, 8), (2, 2, 2)), ((32, 1), (16, 8, 128)))
                    b = TensorLayout(((4, 8), (2, 2)), ((16, 1), (8, 64)))
                    c = TensorLayout(((4, 8), (2, 2)), ((32, 1), (16, 8)))
                    mma_atom = MmaAtom("warp", (16, 8, 16), a, b, c, c, (1, 2))
                    warp_in_threadblock = Level(
                        "warp", "thread_block", (warp_m, warp_n), TensorLayout((warp_m, warp_n)), (repeat_m, repeat_n)
                    )
                    tiled_mma = TiledMma(mma_atom, [warp_in_threadblock])
                    block_m = 16 * warp_m * repeat_m
                    block_n = 16 * warp_n * repeat_n
                    if warp_m * warp_n == 8 and (block_m, block_n) not in compute_bound[:2]:
                        continue
                    if (block_m, block_n) in compute_bound:
                        _tiled_mma_space_1.append(tiled_mma)
                    else:
                        _tiled_mma_space_2.append(tiled_mma)
    _tiled_mma_space_2.extend(_tiled_mma_space_1)


principal_tiled_mma = _tiled_mma_space_1[0]
current_compute_capability: int = 0


def cast_fp16(x: Expr):
    return float16(x)


def cast_bf16(x: Expr):
    return bfloat16(x)


class MatmulF16CuteTask(Task):
    """
    A matmul template that enables f16 tensorcore with CuTe dialect.
    Currently, this is amost a copy-paste of matmul_f16.py with the only change of epilogue.
    We will replace the rest parts with our CuTe dialect later.
    """

    def __init__(
        self,
        a: TensorNode,
        b: TensorNode,
        acc_dtype: Union[DataType, str],
        parallel_k_parts: int = 1,
        transpose_b: bool = False,
    ):
        self.transpose_b = transpose_b

        if not a.type.dtype == b.type.dtype:
            raise ValueError(f'Both inputs must have the same dtype, but got {a.type.dtype} and {b.type.dtype}')

        both_f16 = a.type.dtype == float16
        both_bf16 = a.type.dtype == bfloat16

        if both_f16:
            target_float_type = float16
        elif both_bf16:
            target_float_type = bfloat16
        else:
            raise ValueError(
                f'Both inputs must be float16 or bfloat tensors, but got {a.type.dtype} and {b.type.dtype}'
            )

        self.target_float_type = target_float_type

        if len(a.shape) < 2 or len(b.shape) < 2:
            raise ValueError('Matrix multiplication expect at least 2D tensor, got {} and {}'.format(a.shape, b.shape))

        if not self.transpose_b:
            self._assert(
                a.shape[-1] == b.shape[-2],
                msg=(
                    'Matrix multiplication expect tensor A and B with shape [..., M, K] and [..., K, N]'
                    ', got {} and {}'.format(a.shape, b.shape)
                ),
            )
        else:
            self._assert(
                a.shape[-1] == b.shape[-1],
                msg=(
                    'Matrix multiplication expect tensor A and B with shape [..., M, K] and [..., K, N]'
                    ', got {} and {}'.format(a.shape, b.shape)
                ),
            )

        self._assert(
            can_mutually_broadcast(a.shape[:-2], b.shape[:-2]),
            msg=(
                'Matrix multiplication expect tensor A and B with compatible broadcast shape, '
                'got {} and {}'.format(a.shape, b.shape)
            ),
        )

        a_shape = a.shape
        b_shape = b.shape
        k_size = a.shape[-1]
        if not transpose_b:
            c_shape = [parallel_k_parts] + broadcast_shape(a.shape[:-2], b.shape[:-2]) + [a_shape[-2], b_shape[-1]]
        else:
            c_shape = [parallel_k_parts] + broadcast_shape(a.shape[:-2], b.shape[:-2]) + [a_shape[-2], b_shape[-2]]

        k_part_extent = cdiv(k_size, parallel_k_parts)
        acc_dtype = data_type(acc_dtype)

        def inner_compute(k_part, indices, k):
            if not transpose_b:
                return if_then_else(
                    k_part * k_part_extent + k < k_size,
                    cast(
                        a[broadcast_indices(indices[:-2], a.shape[:-2], c_shape[1:-2]) + [indices[-2], k]]
                        * b[broadcast_indices(indices[:-2], b.shape[:-2], c_shape[1:-2]) + [k, indices[-1]]],
                        acc_dtype,
                    ),
                    acc_dtype(0.0),
                )
            else:
                return if_then_else(
                    k_part * k_part_extent + k < k_size,
                    cast(
                        a[broadcast_indices(indices[:-2], a.shape[:-2], c_shape[1:-2]) + [indices[-2], k]]
                        * b[broadcast_indices(indices[:-2], b.shape[:-2], c_shape[1:-2]) + [indices[-1], k]],
                        acc_dtype,
                    ),
                    acc_dtype(0.0),
                )

        def outer_compute(k_part, *indices):
            return target_float_type(
                reduce(shape=[k_part_extent], fcompute=partial(inner_compute, k_part, indices), reduce_type='sum')
            )

        c = compute(name='c', shape=c_shape, fcompute=outer_compute)

        super().__init__(
            name=f'matmul_{target_float_type.short_name}_pk_cute_transpose_b_{transpose_b}',
            inputs=[a, b],
            outputs=[c],
            attributes={'acc_dtype': acc_dtype, 'parallel_k_parts': parallel_k_parts, 'transpose_b': transpose_b},
        )

    def allow_prologue(self) -> bool:
        return False

    def allow_epilogue(self) -> bool:
        return True

    def implement_cuda(self, working_dir: str) -> List[IRModule]:
        compute_capability = hidet.option.cuda.get_arch_pair()
        compute_capability = compute_capability[0] * 10 + compute_capability[1]
        global current_compute_capability
        current_compute_capability = compute_capability
        return tune.extract_ir_modules(self.schedule)

    def prologue(self):
        # input shapes
        node_a, node_b, node_c = self.inputs[0], self.inputs[1], self.outputs[0]
        a_shape: Tuple[Int, ...] = node_a.shape
        b_shape: Tuple[Int, ...] = node_b.shape
        c_shape: Tuple[Int, ...] = node_c.shape

        transpose_b = self.attrs['transpose_b']
        if not transpose_b:
            m_size, n_size, k_size = a_shape[-2], b_shape[-1], a_shape[-1]
        else:
            m_size, n_size, k_size = a_shape[-2], b_shape[-2], a_shape[-1]

        a_head, b_head, c_head = list(a_shape[:-2]), list(b_shape[:-2]), list(c_shape[:-2])
        return m_size, n_size, k_size, a_head, b_head, c_head

    def _get_max_smem(self):
        smem_limits = {70: 96000, 72: 96000, 75: 64000, 80: 163000, 86: 99000, 87: 163000, 89: 99000, 90: 227000}
        return 99000 if current_compute_capability > 90 else smem_limits[current_compute_capability]

    def _get_store_c(self, n_size: Union[Expr, int], block_m: int, block_n: int, threads: int):
        from hidet.ir.expr import is_constant

        assert is_constant(n_size)
        n_size = int(n_size)
        if n_size % 8 == 0:
            alignment = 8
        elif n_size % 4 == 0:
            alignment = 4
        elif n_size % 2 == 0:
            alignment = 2
        else:
            alignment = 1
        atom_shape = (1, alignment)
        atom = TensorLayout(((1,), (1, alignment)), ((1,), (1, 1)))
        tv_atom = ThrValAtom("thread", atom_shape, atom)
        from hidet.utils.py import gcd

        thread_n = gcd(block_n // alignment, threads)
        thread_m = threads // thread_n
        repeat_n = block_n // (alignment * thread_n)
        repeat_m = block_m // thread_m
        threads_in_thread_block = Level(
            "thread",
            "thread_block",
            (thread_m, thread_n),
            TensorLayout((thread_n, thread_m), (thread_m, 1)),
            (repeat_m, repeat_n),
        )
        return TiledTensorLayout(tv_atom, [threads_in_thread_block])

    @tune.space(2, tiled_mma=_tiled_mma_space_1, block_k=[32, 64], multi_stage=[True, False], use_cublas=[True, False])
    @tune.space(1, tiled_mma=_tiled_mma_space_1[:1], block_k=[32, 64], multi_stage=[True], use_cublas=[False])
    def schedule(self, tiled_mma=principal_tiled_mma, block_k=32, multi_stage=True, use_cublas=False) -> IRModule:
        # pylint: disable=unused-variable
        # For the bfloat16 case, there is no mma config with float16 accumulator, only float32
        target_float_type = self.target_float_type
        if target_float_type == bfloat16:
            acc_dtype = self.attrs['acc_dtype']
            tune.check(acc_dtype == float32, 'bfloat16 only supports float32 accumulator')

        if use_cublas:
            transpose_b = self.attrs['transpose_b']
            # TODO: Is there a way to support cuBLAS with B transposed?
            tune.check(not transpose_b, 'Cublas does not support transpose_b')
            tune.check(multi_stage)
            k_parts = self.attrs['parallel_k_parts']
            # Don't know how to convert the matmuls with parallel_k opt to batched matmul, so we disable it here.
            tune.check(k_parts == 1)
            return self.matmul_cublas(tiled_mma, block_k)
        elif multi_stage:
            return self.matmul_multi_buffer(tiled_mma, block_k)
        else:
            return self.matmul_single_buffer(tiled_mma, block_k)

    def matmul_cublas(self, tiled_mma: TiledMma, block_k: int):
        # Hack to reduce redundant schedules. When use_cublas == False, other tuning params are irrelevant
        # and we only need one copy of the schedule.
        _, c_tv_layout = tiled_mma.c_tv_layout()
        _, principal_c_tv_layout = principal_tiled_mma.c_tv_layout()
        tune.check(c_tv_layout == principal_c_tv_layout)
        tune.check(block_k == 32)

        from hidet.graph.ops.utils.schedule_utils import get_cublas_matmul_schedule
        from hidet.cuda.cublas import cublasComputeType

        acc_dtype = self.attrs['acc_dtype']
        dtype = self.inputs[0].type.dtype
        if acc_dtype == float32:
            compute_type = cublasComputeType.CUBLAS_COMPUTE_32F
        else:
            compute_type = cublasComputeType.CUBLAS_COMPUTE_16F
        node_a, node_b, node_c = self.inputs[0], self.inputs[1], self.outputs[0]
        a_shape: Tuple[Int, ...] = node_a.shape
        b_shape: Tuple[Int, ...] = node_b.shape
        c_shape: Tuple[Int, ...] = node_c.shape
        return get_cublas_matmul_schedule(a_shape, b_shape, c_shape, dtype, dtype, dtype, compute_type)

    def matmul_single_buffer(self, tiled_mma: TiledMma, block_k: int):
        acc_dtype = self.attrs['acc_dtype']
        target_float_type = self.target_float_type
        f16_acc = acc_dtype == target_float_type

        k_parts = self.attrs['parallel_k_parts']
        m_size, n_size, k_size, a_head, b_head, c_head = self.prologue()
        k_part_extent = (k_size + block_k * k_parts - 1) // (block_k * k_parts) * block_k
        a_shape, _ = tiled_mma.a_tv_layout()
        c_shape, c_tv_layout = tiled_mma.c_tv_layout()
        _, inst_k = a_shape
        block_m, block_n = c_shape
        threads = c_tv_layout[0][0].size()

        maximum_smem_bytes = self._get_max_smem()
        dynamic_smem_bytes = (block_m + block_n) * block_k * float16.nbytes
        tune.check(dynamic_smem_bytes <= maximum_smem_bytes, "not enough shared memory")
        store_c_layout = self._get_store_c(n_size, block_m, block_n, threads)
        tiled_copy_c = TiledCopy.from_tiled_tensor_layout(store_c_layout)
        tune.check(block_n <= int(n_size) * 4)

        # at least two buffers
        tune.check(k_part_extent > block_k)
        effective_k_parts = cdiv(k_size, k_part_extent)

        transpose_b = self.attrs['transpose_b']
        with hidet.script_module() as script_module:

            @hidet.script
            def func(
                a: target_float_type[a_head + [m_size, k_size]],
                b_ptr: ~target_float_type,
                c: target_float_type[c_head + [m_size, n_size]],
            ):
                attrs.func_kind = "cuda_kernel"
                attrs.cuda.block_dim = threads
                attrs.cuda.grid_dim = cdiv(m_size, block_m) * cdiv(n_size, block_n), prod(c_head)
                attrs.cuda.dynamic_smem_bytes = 0

                group_size_m = 8
                pid = blockIdx.x
                num_pid_m = cdiv(m_size, block_m)
                num_pid_n = cdiv(n_size, block_n)
                num_pid_in_group = group_size_m * num_pid_n
                group_id = pid // num_pid_in_group
                first_pid_m = group_id * group_size_m
                group_size_m = min(num_pid_m - first_pid_m, group_size_m)
                pid_m = first_pid_m + (pid % group_size_m)
                pid_n = (pid % num_pid_in_group) // group_size_m

                c_head_index = spatial(*c_head).map(blockIdx.y)
                k_part = c_head_index[0]
                k_extent = k_part_extent
                if k_size < k_part_extent * k_parts:
                    if k_part >= effective_k_parts:
                        k_extent = 0
                    elif k_part == effective_k_parts - 1:
                        k_extent = k_size - k_part * k_part_extent
                    else:
                        k_extent = k_part_extent
                k_residue = k_extent % block_k if k_extent % block_k != 0 else block_k
                k_blocks = cdiv(k_extent, block_k)

                # manually annotate the tensor layout to save compile time
                ts_a = make_tensor(target_float_type, TensorLayout((block_m, block_k), (block_k, 1)), "shared")
                if transpose_b:
                    ts_b = make_tensor(target_float_type, TensorLayout((block_n, block_k), (block_k, 1)), "shared")
                else:
                    ts_b = make_tensor(target_float_type, TensorLayout((block_n, block_k), (1, block_n)), "shared")

                tr_a = make_tensor(target_float_type, layout_auto((block_m, inst_k * 2)), "register")
                tr_b = make_tensor(target_float_type, layout_auto((block_n, inst_k * 2)), "register")
                tr_c = make_tensor(acc_dtype, auto_layout, "register")
                fill(tr_c, 0.0)

                a_head_index = broadcast_indices(c_head_index[1:], a_head, c_head[1:])
                b_head_index = broadcast_indices(c_head_index[1:], b_head, c_head[1:])
                tg_a = tensor_view(
                    a[a_head_index][pid_m * block_m :, k_part * k_part_extent :],
                    TensorLayout((block_m, k_part_extent), (k_size, 1)),
                    "global",
                )

                if not transpose_b:
                    b = as_tensor_pointer(b_ptr, target_float_type.name, b_head + [k_size, n_size])
                    gmem_b = b[b_head_index][k_part * k_part_extent :, pid_n * block_n :]
                    tg_b = tensor_view(gmem_b, TensorLayout((block_n, k_part_extent), (1, n_size)), "global")
                else:
                    b = as_tensor_pointer(b_ptr, target_float_type.name, b_head + [n_size, k_size])
                    gmem_b = b[b_head_index][pid_n * block_n :, k_part * k_part_extent :]
                    tg_b = tensor_view(gmem_b, TensorLayout((block_n, k_part_extent), (k_size, 1)), "global")

                txga = partition_src(tg_a, auto_copy())
                txsa = partition_dst(ts_a, auto_copy())

                txgb = partition_src(tg_b, auto_copy())
                txsb = partition_dst(ts_b, auto_copy())

                txSa = partition_src(ts_a, auto_copy())
                txra = partition_dst(tr_a, auto_copy())

                txSb = partition_src(ts_b, auto_copy())
                txrb = partition_dst(tr_b, auto_copy())

                msk_a_0 = mask(auto_copy(), [m_size - pid_m * block_m, i32(k_residue)])
                msk_b_0 = mask(auto_copy(), [n_size - pid_n * block_n, i32(k_residue)])

                msk_a = mask(auto_copy(), [m_size - pid_m * block_m, i32(block_k)])
                msk_b = mask(auto_copy(), [n_size - pid_n * block_n, i32(block_k)])

                k_tiles = block_k // inst_k
                for ko in range(k_blocks - 1):
                    copy(auto_copy((block_m, block_k)), txga[:, :, ko], txsa, msk_a)
                    copy(auto_copy((block_n, block_k)), txgb[:, :, ko], txsb, msk_b)

                    cp_async_wait_all()
                    syncthreads()

                    copy(auto_copy(), txSa[:, :, 0], txra[:, :, 0])
                    copy(auto_copy(), txSb[:, :, 0], txrb[:, :, 0])

                    for ki in grid(k_tiles, attrs="u+"):
                        if ki < k_tiles:
                            copy(auto_copy(), txSa[:, :, ki + 1], txra[:, :, (ki + 1) % 2])
                            copy(auto_copy(), txSb[:, :, ki + 1], txrb[:, :, (ki + 1) % 2])

                        mma(tiled_mma, tr_c, txra[:, :, ki % 2], txrb[:, :, ki % 2], tr_c)
                    syncthreads()

                if k_blocks >= 1:
                    ko = k_blocks - 1
                    if k_residue != block_k:
                        copy(auto_copy((block_m, block_k)), txga[:, :, ko], txsa, msk_a_0)
                        copy(auto_copy((block_n, block_k)), txgb[:, :, ko], txsb, msk_b_0)
                    else:
                        copy(auto_copy((block_m, block_k)), txga[:, :, ko], txsa, msk_a)
                        copy(auto_copy((block_n, block_k)), txgb[:, :, ko], txsb, msk_b)
                    cp_async_wait_all()
                    syncthreads()

                    copy(auto_copy(), txSa[:, :, 0], txra[:, :, 0])
                    copy(auto_copy(), txSb[:, :, 0], txrb[:, :, 0])

                    for ki in grid(k_tiles, attrs="u+"):
                        if ki < k_tiles:
                            copy(auto_copy(), txSa[:, :, ki + 1], txra[:, :, (ki + 1) % 2])
                            copy(auto_copy(), txSb[:, :, ki + 1], txrb[:, :, (ki + 1) % 2])

                        mma(tiled_mma, tr_c, txra[:, :, ki % 2], txrb[:, :, ki % 2], tr_c)
                    syncthreads()

                if f16_acc:
                    tr_C = rearrange(tr_c, store_c_layout, "register")
                else:
                    tr_C = rearrange(cast(tr_c, target_float_type), store_c_layout, "register")

                extents = [m_size - pid_m * block_m, n_size - pid_n * block_n]
                offset_m, offset_n = pid_m * block_m, pid_n * block_n
                c_head_index = spatial(*c_head).map(blockIdx.y)
                collective_store(tiled_copy_c, tr_C, c, c_head_index + [offset_m, offset_n], extents)

        return script_module.ir_module()

    def _get_stages_heuristic(self, block_m, block_n, block_k, target_float_type):
        _stages_dict: Dict[Tuple[int, int], int] = {(128, 256): 3, (256, 128): 3, (128, 128): 4}
        stages = _stages_dict.get((block_m, block_n), None)
        if stages is not None:
            return stages
        else:
            maximum_smem_bytes = self._get_max_smem()
            stages = maximum_smem_bytes // ((block_m + block_n) * block_k * target_float_type.nbytes)
            stages = max(2, min(stages, 6))
            return stages

    def matmul_multi_buffer(self, tiled_mma: TiledMma, block_k: int):
        acc_dtype = self.attrs['acc_dtype']
        target_float_type = self.target_float_type
        f16_acc = acc_dtype == target_float_type

        k_parts = self.attrs['parallel_k_parts']
        m_size, n_size, k_size, a_head, b_head, c_head = self.prologue()
        k_part_extent = (k_size + block_k * k_parts - 1) // (block_k * k_parts) * block_k
        a_shape, _ = tiled_mma.a_tv_layout()
        c_shape, c_tv_layout = tiled_mma.c_tv_layout()
        _, inst_k = a_shape
        block_m, block_n = c_shape
        threads = c_tv_layout[0][0].size()

        maximum_smem_bytes = self._get_max_smem()
        stages = self._get_stages_heuristic(block_m, block_n, block_k, target_float_type)
        dynamic_smem_bytes = stages * (block_m + block_n) * block_k * target_float_type.nbytes
        tune.check(dynamic_smem_bytes <= maximum_smem_bytes, "not enough shared memory")
        store_c_layout = self._get_store_c(n_size, block_m, block_n, threads)
        tiled_copy_c = TiledCopy.from_tiled_tensor_layout(store_c_layout)
        tune.check(block_n <= int(n_size) * 4)

        # at least two buffers
        tune.check(k_part_extent > block_k)
        effective_k_parts = cdiv(k_size, k_part_extent)

        transpose_b = self.attrs['transpose_b']
        with hidet.script_module() as script_module:

            @hidet.script
            def func(
                a: target_float_type[a_head + [m_size, k_size]],
                b_ptr: ~target_float_type,
                c: target_float_type[c_head + [m_size, n_size]],
            ):
                attrs.func_kind = "cuda_kernel"
                attrs.cuda.block_dim = threads
                attrs.cuda.grid_dim = cdiv(m_size, block_m) * cdiv(n_size, block_n), prod(c_head)
                attrs.cuda.dynamic_smem_bytes = 0

                group_size_m = 8
                pid = blockIdx.x
                num_pid_m = cdiv(m_size, block_m)
                num_pid_n = cdiv(n_size, block_n)
                num_pid_in_group = group_size_m * num_pid_n
                group_id = pid // num_pid_in_group
                first_pid_m = group_id * group_size_m
                group_size_m = min(num_pid_m - first_pid_m, group_size_m)
                pid_m = first_pid_m + (pid % group_size_m)
                pid_n = (pid % num_pid_in_group) // group_size_m

                # manually annotate the tensor layout to save compile time
                ts_a = make_tensor(
                    target_float_type,
                    TensorLayout((block_m, block_k, stages), (block_k, 1, block_m * block_k)),
                    "shared",
                )
                if transpose_b:
                    ts_b = make_tensor(
                        target_float_type,
                        TensorLayout((block_n, block_k, stages), (block_k, 1, block_n * block_k)),
                        "shared",
                    )
                else:
                    ts_b = make_tensor(
                        target_float_type,
                        TensorLayout((block_n, block_k, stages), (1, block_n, block_n * block_k)),
                        "shared",
                    )

                tr_a = make_tensor(target_float_type, layout_auto((block_m, inst_k * 2)), "register")
                tr_b = make_tensor(target_float_type, layout_auto((block_n, inst_k * 2)), "register")
                tr_c = make_tensor(acc_dtype, auto_layout, "register")
                fill(tr_c, 0.0)

                c_head_index = spatial(*c_head).map(blockIdx.y)
                k_part = c_head_index[0]
                k_extent = k_part_extent
                if k_size < k_part_extent * k_parts:
                    if k_part >= effective_k_parts:
                        k_extent = 0
                    elif k_part == effective_k_parts - 1:
                        k_extent = k_size - k_part * k_part_extent
                    else:
                        k_extent = k_part_extent
                k_residue = k_extent % block_k if k_extent % block_k != 0 else block_k
                k_blocks = cdiv(k_extent, block_k)

                a_head_index = broadcast_indices(c_head_index[1:], a_head, c_head[1:])
                b_head_index = broadcast_indices(c_head_index[1:], b_head, c_head[1:])
                tg_a = tensor_view(
                    a[a_head_index][pid_m * block_m :, k_part * k_part_extent :],
                    TensorLayout((block_m, k_part_extent), (k_size, 1)),
                    "global",
                )
                if not transpose_b:
                    b = as_tensor_pointer(b_ptr, target_float_type.name, b_head + [k_size, n_size])
                    gmem_b = b[b_head_index][k_part * k_part_extent :, pid_n * block_n :]
                    tg_b = tensor_view(gmem_b, TensorLayout((block_n, k_part_extent), (1, n_size)), "global")
                else:
                    b = as_tensor_pointer(b_ptr, target_float_type.name, b_head + [n_size, k_size])
                    gmem_b = b[b_head_index][pid_n * block_n :, k_part * k_part_extent :]
                    tg_b = tensor_view(gmem_b, TensorLayout((block_n, k_part_extent), (k_size, 1)), "global")

                txga = partition_src(tg_a, auto_copy())
                txsa = partition_dst(ts_a, auto_copy())

                txgb = partition_src(tg_b, auto_copy())
                txsb = partition_dst(ts_b, auto_copy())

                msk_a_0 = mask(auto_copy(), [m_size - pid_m * block_m, i32(k_residue)])
                msk_b_0 = mask(auto_copy(), [n_size - pid_n * block_n, i32(k_residue)])

                msk_a = mask(auto_copy(), [m_size - pid_m * block_m, i32(block_k)])
                msk_b = mask(auto_copy(), [n_size - pid_n * block_n, i32(block_k)])

                for s in range(stages - 1):
                    if s < k_blocks:
                        if k_residue != block_k and s == k_blocks - 1:
                            copy(auto_copy((block_m, block_k)), txga[:, :, s], txsa[:, :, s], msk_a_0)
                            copy(auto_copy((block_n, block_k)), txgb[:, :, s], txsb[:, :, s], msk_b_0)
                        else:
                            copy(auto_copy((block_m, block_k)), txga[:, :, s], txsa[:, :, s], msk_a)
                            copy(auto_copy((block_n, block_k)), txgb[:, :, s], txsb[:, :, s], msk_b)
                    cp_async_commit_group()
                cp_async_wait_group(allow_on_fly_groups=stages - 2)
                syncthreads()

                smem_pipe_read = 0
                smem_pipe_write = stages - 1

                txSa = partition_src(ts_a, auto_copy())
                txra = partition_dst(tr_a, auto_copy())

                txSb = partition_src(ts_b, auto_copy())
                txrb = partition_dst(tr_b, auto_copy())

                txSa_p = txSa[:, :, :, smem_pipe_read]
                txSb_p = txSb[:, :, :, smem_pipe_read]

                copy(auto_copy(), txSa_p[:, :, 0], txra[:, :, 0])
                copy(auto_copy(), txSb_p[:, :, 0], txrb[:, :, 0])

                k_tiles = block_k // inst_k
                for ko in range(k_blocks):
                    for ki in grid(k_tiles, attrs="u+"):
                        if ki == k_tiles - 1:
                            cp_async_wait_group(allow_on_fly_groups=stages - 2)
                            syncthreads()

                        k_tile_next = (ki + 1) % k_tiles
                        copy(auto_copy(), txSa[:, :, k_tile_next, smem_pipe_read], txra[:, :, (ki + 1) % 2])
                        copy(auto_copy(), txSb[:, :, k_tile_next, smem_pipe_read], txrb[:, :, (ki + 1) % 2])
                        if ki == 0:
                            if ko + stages - 1 < k_blocks:
                                if k_residue != block_k and ko + stages == k_blocks:
                                    copy(
                                        auto_copy((block_m, block_k)),
                                        txga[:, :, ko + stages - 1],
                                        txsa[:, :, smem_pipe_write],
                                        msk_a_0,
                                    )
                                    copy(
                                        auto_copy((block_n, block_k)),
                                        txgb[:, :, ko + stages - 1],
                                        txsb[:, :, smem_pipe_write],
                                        msk_b_0,
                                    )
                                else:
                                    copy(
                                        auto_copy((block_m, block_k)),
                                        txga[:, :, ko + stages - 1],
                                        txsa[:, :, smem_pipe_write],
                                        msk_a,
                                    )
                                    copy(
                                        auto_copy((block_n, block_k)),
                                        txgb[:, :, ko + stages - 1],
                                        txsb[:, :, smem_pipe_write],
                                        msk_b,
                                    )
                            smem_pipe_write = smem_pipe_read
                            cp_async_commit_group()

                        if ki == k_tiles - 2:
                            smem_pipe_read += 1
                            smem_pipe_read = 0 if smem_pipe_read == stages else smem_pipe_read

                        mma(tiled_mma, tr_c, txra[:, :, ki % 2], txrb[:, :, ki % 2], tr_c)

                if f16_acc:
                    tr_C = rearrange(tr_c, store_c_layout, "register")
                else:
                    tr_C = rearrange(cast(tr_c, target_float_type), store_c_layout, "register")

                extents = [m_size - pid_m * block_m, n_size - pid_n * block_n]
                offset_m, offset_n = pid_m * block_m, pid_n * block_n
                c_head_index = spatial(*c_head).map(blockIdx.y)
                collective_store(tiled_copy_c, tr_C, c, c_head_index + [offset_m, offset_n], extents)

        return script_module.ir_module()


class MatmulF16CuteOp(Operator):
    def __init__(
        self, a: Tensor, b: Tensor, acc_dtype: Union[DataType, str], parallel_k_parts=1, transpose_b: bool = False
    ):
        if not (isinstance(parallel_k_parts, int) and not isinstance(parallel_k_parts, bool)):
            raise ValueError('parallel_k_parts must be an integer, got {}'.format(parallel_k_parts))
        acc_dtype = data_type(acc_dtype)
        super().__init__(
            inputs=[a, b],
            attributes={'acc_dtype': acc_dtype, 'parallel_k_parts': parallel_k_parts, 'transpose_b': transpose_b},
            task=MatmulF16CuteTask(input_like(a, 'a'), input_like(b, 'b'), acc_dtype, parallel_k_parts, transpose_b),
        )


def matmul_f16_cute(
    a: Tensor, b: Tensor, parallel_k_parts=1, acc_dtype: Union[DataType, str] = "float32", transpose_b: bool = False
) -> Tensor:
    if len(a.shape) < 2 or len(b.shape) < 2:
        raise ValueError('a and b must have at least 2 dimensions, got shape {} and {}'.format(a.shape, b.shape))
    # TODO: impliment dynamic run-time shape assertion
    if not (isinstance(a.shape[-1], Expr) or isinstance(b.shape[-1], Expr)) and (
        a.shape[-1] % 2 != 0 or b.shape[-1] % 2 != 0
    ):
        raise ValueError('Expect the last dimension of the input tensors to be a multiple of 2')
    if a.dtype != b.dtype:
        raise ValueError('a and b must have the same dtype, got {} and {}'.format(a.dtype, b.dtype))

    valid_dtypes = [dtypes.float16, dtypes.bfloat16]
    if a.dtype not in valid_dtypes or b.dtype not in valid_dtypes:
        raise ValueError('matmul_f16_cute only supports float16 or bfloat16, got {} and {}'.format(a.dtype, b.dtype))

    acc_dtype = data_type(acc_dtype)
    return MatmulF16CuteOp(a, b, acc_dtype, parallel_k_parts, transpose_b).outputs[0]
