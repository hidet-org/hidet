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
"""
A matmul template that enables f16 tensorcore with CuTe dialect.
The main part of this kernel is implemented with Hidet script, but we rewrite the epilogue with
operations in CuTe dialect, which enables us to coalesce and vectorize the memory access in the
epilogue.
"""
from typing import List, Tuple, Union

from hidet.ir.cute import auto_copy, auto_layout
from hidet.ir.cute.ops import make_tensor, fill, mask
from hidet.ir.primitives.runtime import request_cuda_workspace
from hidet.ir.type import DataType, data_type
from hidet.ir.dtypes import float16, float32, bfloat16
from hidet.ir.expr import Int, Expr, is_constant
from hidet.ir.func import Function
from hidet.ir.module import IRModule
from hidet.ir.compute import TensorNode
from hidet.ir.task import Task

from hidet.graph.ops.utils import input_like, can_mutually_broadcast
from hidet.ir.library import tune
from hidet.graph.operator import Operator, Tensor
from hidet.utils.py import is_power_of_two, cdiv, prod
from hidet.graph.ops.utils import broadcast_indices

from hidet.option import get_parallel_k


def get_parallel_k_candidates(level: int):
    parallel_k_option = get_parallel_k()

    if isinstance(parallel_k_option, int):
        candidates = [parallel_k_option]
    elif parallel_k_option == 'default':
        candidates = [0]
    elif parallel_k_option == 'disabled':
        candidates = [1]
    elif parallel_k_option == 'search':
        candidates_map = {0: [1], 1: [1, 2], 2: [1, 2, 4, 8]}
        candidates = candidates_map[level]
    else:
        raise ValueError(f'Unsupported parallel_k_option: {parallel_k_option}')
    for candidate in candidates:
        yield candidate


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

    def __init__(self, a: TensorNode, b: TensorNode, acc_dtype: Union[DataType, str], transpose_b: bool = False):
        from hidet.ir.compute import cops

        self.transpose_b = transpose_b

        if not a.type.dtype == b.type.dtype:
            raise ValueError(f'Both inputs must have the same dtype, but got {a.type.dtype} and {b.type.dtype}')

        if a.type.dtype.is_any_float16():
            target_float_type = a.type.dtype
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

        acc_dtype = data_type(acc_dtype)

        c = cops.matmul(a, b, tb=transpose_b)

        super().__init__(
            name=f'matmul_{target_float_type.short_name}_pk_cute_transpose_b_{transpose_b}',
            inputs=[a, b],
            outputs=[c],
            attributes={'acc_dtype': acc_dtype, 'transpose_b': transpose_b},
        )

    def allow_prologue(self) -> bool:
        return False

    def allow_epilogue(self) -> bool:
        return True

    def implement_cuda(self, working_dir: str) -> List[IRModule]:
        return tune.extract_ir_modules(self.schedule)

    def get_alignment(self, last_dim: int) -> int:
        if last_dim % 8 == 0:
            return 8
        elif last_dim % 4 == 0:
            return 4
        elif last_dim % 2 == 0:
            return 2
        else:
            assert False, "Invalid alignment, please always pack 2 fp16 elements"

    @tune.space(
        2,
        block_m=[32, 64, 128, 256],
        block_n=[32, 64, 128, 256],
        block_k=[8, 16, 32, 64, 128],
        warp_m=[16, 32, 48, 64],
        warp_n=[16, 32, 48, 64],
        warp_k=[8, 16, 32, 64],
        mma=['m16n8k16'],
        parallel_k_parts=get_parallel_k_candidates,
        use_cublas=[True, False],
    )
    @tune.space(
        1,
        block_m=[128],
        block_n=[128],
        block_k=[16],
        warp_m=[64],
        warp_n=[64],
        warp_k=[16],
        mma=['m16n8k16'],
        parallel_k_parts=get_parallel_k_candidates,
        use_cublas=[True, False],
    )
    def schedule(
        self,
        block_m=64,
        block_n=128,
        block_k=16,
        warp_m=32,
        warp_n=64,
        warp_k=16,
        mma: str = 'm16n8k16',
        parallel_k_parts=1,
        use_cublas=False,
    ) -> IRModule:
        # pylint: disable=unused-variable
        import hidet
        from hidet.ir.type import tensor_type
        from hidet.lang import attrs, col_spatial, view, u32, tensor_pointer, grid
        from hidet.lang.layout import row_major
        from hidet.lang.mapping import spatial, auto_map
        from hidet.lang.cuda import blockIdx, threadIdx, syncthreads, dynamic_shared_memory
        from hidet.lang.cuda import MmaConfig, mma_sync, cp_async, cp_async_wait_all, ldmatrix
        from hidet.lang.cuda import register_tensor
        from hidet.lang.constructs.declare import as_tensor_pointer

        transpose_b = self.attrs['transpose_b']
        target_float_type = self.target_float_type
        cast_func = cast_fp16 if target_float_type == float16 else cast_bf16

        # input shapes
        node_a, node_b, node_c = self.inputs[0], self.inputs[1], self.outputs[0]
        a_shape: List[Expr] = node_a.shape
        b_shape: List[Expr] = node_b.shape
        c_shape: List[Expr] = node_c.shape

        if not transpose_b:
            m_size, n_size, k_size = a_shape[-2], b_shape[-1], a_shape[-1]
        else:
            m_size, n_size, k_size = a_shape[-2], b_shape[-2], a_shape[-1]

        c_head_non_parallel_k = list(c_shape[:-2])

        if self.has_symbolic_shape():
            tune.check(parallel_k_parts == 1, 'parallel_k_parts must be 1 with symbolic shape')

        if parallel_k_parts == 0:
            # Hack: 0 is used to indicate that the `default` parallel_k strategy will be used
            batch_size = prod(c_shape[:-2])
            if is_constant(batch_size, m_size):
                estimate_blocks = batch_size * cdiv(m_size, 64) * cdiv(n_size, 64)
                estimate_concurrent_blocks = 80 * 5
                max_k_parts = cdiv(k_size, 64)
                k_parts = min(cdiv(estimate_concurrent_blocks, estimate_blocks), max_k_parts)
                parallel_k_parts = k_parts

        assert parallel_k_parts > 0, 'parallel_k_parts must be greater than 0'
        parallel_c_shape = [parallel_k_parts] + c_shape
        a_head, b_head, c_head = list(a_shape[:-2]), list(b_shape[:-2]), list(parallel_c_shape[:-2])
        c_head_matmul = c_head if parallel_k_parts > 1 else c_head_non_parallel_k
        k_part_extent = cdiv(cdiv(k_size, parallel_k_parts), 8) * 8
        acc_dtype = self.attrs['acc_dtype']

        if use_cublas:
            from hidet.graph.ops.utils.schedule_utils import get_cublas_matmul_schedule
            from hidet.cuda.cublas import cublasComputeType

            dtype = self.inputs[0].type.dtype
            if acc_dtype == float32:
                compute_type = cublasComputeType.CUBLAS_COMPUTE_32F
            else:
                compute_type = cublasComputeType.CUBLAS_COMPUTE_16F
            # Hack to reduce redundant schedules. When use_cublas == False, other tuning params are irrelevant
            # and we only need one copy of the schedule.
            schedule_filter = (
                block_m == 128
                and block_n == 128
                and block_k == 16
                and warp_m == 64
                and warp_n == 64
                and warp_k == 16
                and mma == 'm16n8k16'
            )
            tune.check(schedule_filter)
            tune.check(parallel_k_parts == 1)
            return get_cublas_matmul_schedule(
                a_shape, b_shape, c_shape, dtype, dtype, dtype, compute_type, transpose_b=transpose_b
            )

        # schedule parameters

        # For the bfloat16 case, there is no mma config with float16 accumulator, only float32
        if target_float_type == bfloat16:
            tune.check(acc_dtype == float32, 'bfloat16 only supports float32 accumulator')

        mma_configs_f16 = {'m16n8k8': MmaConfig.m16n8k8_f16_f16(), 'm16n8k16': MmaConfig.m16n8k16_f16_f16()}
        mma_configs_f32 = (
            {'m16n8k8': MmaConfig.m16n8k8_f16_f32(), 'm16n8k16': MmaConfig.m16n8k16_f16_f32()}
            if target_float_type == float16
            else {'m16n8k8': MmaConfig.m16n8k8_bf16_f32(), 'm16n8k16': MmaConfig.m16n8k16_bf16_f32()}
        )
        tune.check(mma in mma_configs_f16 or mma in mma_configs_f32)
        mma_config = mma_configs_f16[mma] if acc_dtype == float16 else mma_configs_f32[mma]

        mma_m, mma_n, mma_k = mma_config.m, mma_config.n, mma_config.k  # 16, 8, 16
        warp_count_m, warp_count_n, warp_count_k = block_m // warp_m, block_n // warp_n, block_k // warp_k
        mma_count_m, mma_count_n, mma_count_k = warp_m // mma_m, warp_n // mma_n, warp_k // mma_k
        threads = warp_count_m * warp_count_n * warp_count_k * 32
        grid_dim: Tuple[Int, Int, Int] = cdiv(m_size, block_m), cdiv(n_size, block_n), prod(c_head)
        dynamic_smem_bytes = max(2 * (block_m + block_n) * block_k * 2, block_m * block_n * acc_dtype.nbytes)

        tune.check(block_m % warp_m == block_n % warp_n == block_k % warp_k == 0, 'warp dims divide block dims')
        tune.check(warp_m % mma_m == warp_n % mma_n == warp_k % mma_k == 0, 'mma dims divide warp dims')
        tune.check(threads <= 1024, 'threads in a block <= 1024')
        maximum_smem_bytes = 49152
        tune.check(dynamic_smem_bytes <= maximum_smem_bytes, 'dynamic shared memory <= 49152')

        if not transpose_b:
            tune.check(block_n % 64 == 0, 'block_n must be multiple of 64, required by async gmem -> smem loading')
        tune.check(block_k % 8 == 0)
        tune.check(is_power_of_two(block_k // 8))
        smem_a_type = tensor_type(
            self.target_float_type.name,
            shape=[block_m, block_k],
            layout=row_major(block_m, block_k // 8).swizzle(1) * row_major(1, 8),
        )
        if not transpose_b:
            smem_b_type = tensor_type(
                target_float_type.name,
                shape=[block_k, block_n],
                layout=row_major(block_k // 8, block_n // 64) * row_major(8, 8).swizzle(1) * row_major(1, 8),
            )
        else:
            smem_b_type = tensor_type(
                target_float_type.name,
                shape=[block_n, block_k],
                layout=row_major(block_n, block_k // 8).swizzle(1) * row_major(1, 8),
            )

        load_smem_a_map = auto_map(block_m, block_k // 8, workers=threads, on_fail=lambda msg: tune.check(False, msg))

        if not transpose_b:
            load_smem_b_map = auto_map(
                block_k, block_n // 8, workers=threads, on_fail=lambda msg: tune.check(False, msg)
            )
        else:
            load_smem_b_map = auto_map(
                block_n, block_k // 8, workers=threads, on_fail=lambda msg: tune.check(False, msg)
            )

        with hidet.script_module() as module:

            @hidet.script
            def load_regs_a(mi: int, k1: int, smem_a: smem_a_type, regs_a: target_float_type[mma_config.a_elements]):
                warp_id, lane_id = threadIdx.x / 32, threadIdx.x % 32
                for wi, wj, wk in spatial(warp_count_m, warp_count_n, warp_count_k).on(warp_id):
                    p, q = col_spatial(16, 2).map(lane_id)
                    row_addr = ~smem_a[wi * warp_m + mi * mma_m + p, wk * warp_k + k1 * mma_k + q * 8]
                    b32_regs = view(regs_a, u32[4])
                    ldmatrix(
                        regs=[b32_regs[0], b32_regs[1], b32_regs[2], b32_regs[3]],
                        smem_addr=row_addr,
                        shared_space_addr=False,
                        trans=False,
                    )

            @hidet.script
            def load_regs_b_2x(
                mj: int, k1: int, smem_b: smem_b_type, regs_b: target_float_type[2 * mma_config.b_elements]
            ):
                """
                We merge two ldmatrix.x2 insts to a single ldmatrix.x4 so that we can improve the throughput.
                """
                warp_id, lane_id = threadIdx.x / 32, threadIdx.x % 32
                if not transpose_b:
                    for wi, wj, wk in spatial(warp_count_m, warp_count_n, warp_count_k).on(warp_id):
                        p, q = col_spatial(16, 2).map(lane_id)
                        row_addr = ~smem_b[wk * warp_k + k1 * mma_k + p, wj * warp_n + mj * mma_n + q * mma_n]
                        regs = view(regs_b, u32[4])
                        ldmatrix(regs=[regs[0], regs[1], regs[2], regs[3]], smem_addr=row_addr, trans=True)
                else:
                    for wi, wj, wk in spatial(warp_count_m, warp_count_n, warp_count_k).on(warp_id):
                        # Need to change the mapping from col_spatial(16, 2) to col_spatial(8, 2, 2)
                        # due to the matrix B's layout of the fragments held by different threads:
                        # https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#mma-16816-b-f16
                        p, q, r = col_spatial(8, 2, 2).map(lane_id)

                        row_addr = ~smem_b[wj * warp_n + mj * mma_n + p + r * 8, wk * warp_k + k1 * mma_k + q * 8]

                        regs = view(regs_b, u32[4])
                        ldmatrix(
                            regs=[regs[0], regs[1], regs[2], regs[3]],
                            smem_addr=row_addr,
                            shared_space_addr=False,
                            trans=False,
                        )

            @hidet.script
            def warp_mma(
                regs_a: target_float_type[mma_config.a_elements],
                regs_b: target_float_type[mma_config.b_elements],
                regs_c: acc_dtype[mma_config.c_elements],
            ):
                mma_sync(mma_config, regs_a, regs_b, regs_c)

            @hidet.script
            def load_smem_a(k0: int, a: target_float_type[a_head + [m_size, k_size]], smem_a: smem_a_type):
                c_head_index = spatial(*c_head).map(blockIdx.z)
                offset_m = blockIdx.x * block_m
                offset_k = c_head_index[0] * k_part_extent + k0 * block_k
                maximum_k = min(k_size, (c_head_index[0] + 1) * k_part_extent)
                gmem_a = a[broadcast_indices(c_head_index[1:], a_head, c_head[1:])][offset_m:, offset_k:]
                for i, k_seg in load_smem_a_map.on(threadIdx.x):
                    k = k_seg * 8
                    src_size = (
                        0
                        if (offset_m + i >= m_size or offset_k + k >= maximum_k)
                        else min(maximum_k - (offset_k + k), 8)
                    )
                    if a_shape[-1] % 8 == 0:
                        cp_async(~smem_a[i, k], ~gmem_a[i, k], cp_size=16, src_size=src_size * 2, cache_level='global')
                    # trivially support other cp_sizes, perhaps do this in a more clever way?
                    elif a_shape[-1] % 4 == 0:
                        cp_async(~smem_a[i, k], ~gmem_a[i, k], cp_size=8, src_size=min(8, src_size * 2))
                        cp_async(~smem_a[i, k + 4], ~gmem_a[i, k + 4], cp_size=8, src_size=max(0, src_size * 2 - 8))
                    elif a_shape[-1] % 2 == 0:
                        cp_async(~smem_a[i, k], ~gmem_a[i, k], cp_size=4, src_size=min(4, src_size * 2))
                        cp_async(
                            ~smem_a[i, k + 2], ~gmem_a[i, k + 2], cp_size=4, src_size=min(4, max(0, src_size * 2 - 4))
                        )
                        cp_async(
                            ~smem_a[i, k + 4], ~gmem_a[i, k + 4], cp_size=4, src_size=min(4, max(0, src_size * 2 - 8))
                        )
                        cp_async(
                            ~smem_a[i, k + 6], ~gmem_a[i, k + 6], cp_size=4, src_size=min(4, max(0, src_size * 2 - 12))
                        )

            @hidet.script
            def load_smem_b(k0: int, b_ptr: ~target_float_type, smem_b: smem_b_type):
                c_head_index = spatial(*c_head).map(blockIdx.z)
                offset_n = blockIdx.y * block_n
                offset_k = c_head_index[0] * k_part_extent + k0 * block_k
                maximum_k = min(k_size, (c_head_index[0] + 1) * k_part_extent)

                if not transpose_b:
                    b = as_tensor_pointer(b_ptr, target_float_type.name, b_head + [k_size, n_size])
                    gmem_b = b[broadcast_indices(c_head_index[1:], b_head, c_head[1:])][offset_k:, offset_n:]
                else:
                    b = as_tensor_pointer(b_ptr, target_float_type.name, b_head + [n_size, k_size])
                    gmem_b = b[broadcast_indices(c_head_index[1:], b_head, c_head[1:])][offset_n:, offset_k:]

                if not transpose_b:
                    for k, j_seg in load_smem_b_map.on(threadIdx.x):
                        j = j_seg * 8
                        src_size = (
                            0
                            if (offset_k + k >= maximum_k or offset_n + j >= n_size)
                            else min(n_size - (offset_n + j), 8)
                        )
                        if b_shape[-1] % 8 == 0:
                            cp_async(
                                ~smem_b[k, j], ~gmem_b[k, j], cp_size=16, src_size=src_size * 2, cache_level='global'
                            )
                        # trivially support other cp_sizes, perhaps do this in a more clever way?
                        elif b_shape[-1] % 4 == 0:
                            cp_async(~smem_b[k, j], ~gmem_b[k, j], cp_size=8, src_size=min(8, src_size * 2))
                            cp_async(~smem_b[k, j + 4], ~gmem_b[k, j + 4], cp_size=8, src_size=max(0, src_size * 2 - 8))
                        elif b_shape[-1] % 2 == 0:
                            cp_async(~smem_b[k, j], ~gmem_b[k, j], cp_size=4, src_size=min(4, src_size * 2))
                            cp_async(
                                ~smem_b[k, j + 2],
                                ~gmem_b[k, j + 2],
                                cp_size=4,
                                src_size=min(4, max(0, src_size * 2 - 4)),
                            )
                            cp_async(
                                ~smem_b[k, j + 4],
                                ~gmem_b[k, j + 4],
                                cp_size=4,
                                src_size=min(4, max(0, src_size * 2 - 8)),
                            )
                            cp_async(
                                ~smem_b[k, j + 6],
                                ~gmem_b[k, j + 6],
                                cp_size=4,
                                src_size=min(4, max(0, src_size * 2 - 12)),
                            )
                else:
                    for j, k_seg in load_smem_b_map.on(threadIdx.x):
                        k = k_seg * 8
                        src_size = (
                            0
                            if (offset_k + k >= maximum_k or offset_n + j >= n_size)
                            else min(maximum_k - (offset_k + k), 8)
                        )
                        if b_shape[-1] % 8 == 0:
                            cp_async(
                                ~smem_b[j, k], ~gmem_b[j, k], cp_size=16, src_size=src_size * 2, cache_level='global'
                            )
                        # trivially support other cp_sizes, perhaps do this in a more clever way?
                        elif b_shape[-1] % 4 == 0:
                            cp_async(~smem_b[j, k], ~gmem_b[j, k], cp_size=8, src_size=min(8, src_size * 2))
                            cp_async(~smem_b[j, k + 4], ~gmem_b[j, k + 4], cp_size=8, src_size=max(0, src_size * 2 - 8))
                        elif b_shape[-1] % 2 == 0:
                            cp_async(~smem_b[j, k], ~gmem_b[j, k], cp_size=4, src_size=min(4, src_size * 2))
                            cp_async(
                                ~smem_b[j, k + 2],
                                ~gmem_b[j, k + 2],
                                cp_size=4,
                                src_size=min(4, max(0, src_size * 2 - 4)),
                            )
                            cp_async(
                                ~smem_b[j, k + 4],
                                ~gmem_b[j, k + 4],
                                cp_size=4,
                                src_size=min(4, max(0, src_size * 2 - 8)),
                            )
                            cp_async(
                                ~smem_b[j, k + 6],
                                ~gmem_b[j, k + 6],
                                cp_size=4,
                                src_size=min(4, max(0, src_size * 2 - 12)),
                            )

            tune.check(mma in ("m16n8k8", "m16n8k16"))
            from hidet.ir.cute import TensorLayout, ThrValAtom, Level, TiledTensorLayout
            from hidet.ir.cute.ops import tensor_view, rearrange, partition_src, partition_dst, copy, arithmetic
            from hidet.ir.cute.collective import collective_store

            atom_shape = (mma_m, mma_n)
            atom = TensorLayout(((4, 8), (2, 2)), ((32, 1), (16, 8)))
            repeat_shape = (warp_m // mma_m, warp_n // mma_n)
            repeat_layout = TensorLayout((repeat_shape[1], repeat_shape[0]), (repeat_shape[0], 1))
            tv_atom = ThrValAtom("warp", atom_shape, atom, repeat_shape, repeat_layout)
            warps_in_threadblock = Level(
                "warp",
                "thread_block",
                (warp_count_m, warp_count_n),
                TensorLayout((warp_count_n, warp_count_m), (warp_count_m, 1)),
            )
            mma_layout = TiledTensorLayout(tv_atom, [warps_in_threadblock])

            alignment = self.get_alignment(int(n_size))
            store_c_map = auto_map(warp_m, warp_n // alignment, workers=32)
            repeat = store_c_map.outer.task_shape
            spatial_shape = store_c_map.inner.task_shape
            atom_shape = (1, alignment)
            atom = TensorLayout(((1), (1, alignment)), ((1), (1, 1)))
            atom = ThrValAtom("thread", atom_shape, atom)
            threads_in_warp = Level(
                "thread",
                "warp",
                spatial_shape,
                TensorLayout((spatial_shape[1], spatial_shape[0]), (spatial_shape[0], 1)),
                repeat,
                TensorLayout(repeat),
            )
            store_c_layout = TiledTensorLayout(atom, [threads_in_warp, warps_in_threadblock])

            from hidet.ir.cute import CopyAtom, TiledCopy

            copy_atom = CopyAtom("warp", store_c_layout.atom.shape, store_c_layout.atom.layout)
            copy_c = TiledCopy(copy_atom, store_c_layout.levels)
            fp16_acc = acc_dtype is float16

            @hidet.script
            def store_c_reg2gmem(
                regs_c: acc_dtype[mma_count_m, mma_count_n, mma_config.c_elements],
                c: target_float_type[c_head_matmul + [m_size, n_size]],
            ):
                t_regs_c = tensor_view(regs_c, mma_layout, "register")
                if fp16_acc:
                    cvt_t_regs_c = rearrange(t_regs_c, store_c_layout, "register")
                else:
                    cvt_t_regs_c = rearrange(arithmetic(t_regs_c, op=cast_func), store_c_layout, "register")
                extents = [m_size - blockIdx.x * block_m, n_size - blockIdx.y * block_n]
                offset_m, offset_n = blockIdx.x * block_m, blockIdx.y * block_n

                if parallel_k_parts > 1:
                    c_head_index = spatial(*c_head).map(blockIdx.z)
                    txrcvt = partition_src(cvt_t_regs_c, copy_c)
                    tg_c = tensor_view(
                        c[c_head_index][offset_m:, offset_n:], TensorLayout((block_m, block_n), (n_size, 1)), "global"
                    )
                    txgc = partition_dst(tg_c, copy_c)
                    mask_c = mask(copy_c, extents)
                    copy(copy_c, txrcvt, txgc, mask_c)
                else:
                    c_head_index = spatial(*c_head_non_parallel_k).map(blockIdx.z)
                    collective_store(copy_c, cvt_t_regs_c, c, c_head_index + [offset_m, offset_n], extents)

            alignment = self.get_alignment(int(n_size))
            tasks = block_m * (block_n // alignment)
            workers = min(tasks, threads)
            store_c_map = auto_map(block_m, block_n // alignment, workers=workers)
            repeat = store_c_map.outer.task_shape
            spatial_shape = store_c_map.inner.task_shape
            atom_shape = (1, alignment)
            atom = TensorLayout(((1), (1, alignment)), ((1), (1, 1)))
            atom = ThrValAtom("thread", atom_shape, atom)
            threads_in_threadblock = Level(
                "thread",
                "thread_block",
                spatial_shape,
                TensorLayout((spatial_shape[1], spatial_shape[0]), (spatial_shape[0], 1)),
                repeat,
                TensorLayout(repeat),
            )
            store_c_layout = TiledTensorLayout(atom, [threads_in_threadblock])
            tiled_copy = TiledCopy.from_tiled_tensor_layout(store_c_layout)
            smem_layout = TensorLayout((block_m, block_n), (block_n, 1))

            @hidet.script
            def store_c_smem2gmem(
                smem_c: acc_dtype[block_m, block_n], c: target_float_type[c_head_matmul + [m_size, n_size]]
            ):
                regs_c = register_tensor(acc_dtype, [store_c_layout.val_layout().size()])
                t_regs_c = tensor_view(regs_c, store_c_layout, "register")
                t_smem_c = tensor_view(smem_c, smem_layout, "shared", volatile=True)
                tcsc = partition_src(t_smem_c, tiled_copy)
                tcrc = partition_dst(t_regs_c, tiled_copy)
                copy(tiled_copy, tcsc, tcrc)
                syncthreads()
                extents = [m_size - blockIdx.x * block_m, n_size - blockIdx.y * block_n]
                offset_m, offset_n = blockIdx.x * block_m, blockIdx.y * block_n
                if parallel_k_parts == 1:
                    c_head_index = spatial(*c_head_non_parallel_k).map(blockIdx.z)
                    if fp16_acc:
                        collective_store(tiled_copy, t_regs_c, c, c_head_index + [offset_m, offset_n], extents)
                    else:
                        collective_store(
                            tiled_copy,
                            arithmetic(t_regs_c, op=cast_func),
                            c,
                            c_head_index + [offset_m, offset_n],
                            extents,
                        )
                else:
                    c_head_index = spatial(*c_head).map(blockIdx.z)
                    tg_c = tensor_view(
                        c[c_head_index][offset_m:, offset_n:], TensorLayout((block_m, block_n), (n_size, 1)), "global"
                    )
                    if fp16_acc:
                        txrcvt = partition_src(t_regs_c, tiled_copy)
                    else:
                        txrcvt = partition_src(arithmetic(t_regs_c, op=cast_func), tiled_copy)
                    txgc = partition_dst(tg_c, tiled_copy)
                    mask_c = mask(tiled_copy, extents)
                    copy(tiled_copy, txrcvt, txgc, mask_c)

            @hidet.script
            def matmul_f16_kernel(
                a: target_float_type[a_head + [m_size, k_size]],
                b_ptr: ~target_float_type,
                c: target_float_type[c_head_matmul + [m_size, n_size]],
            ):
                # matrix multiplication, using mma instruction
                attrs.cuda.grid_dim = grid_dim
                attrs.cuda.block_dim = threads
                # the second 2 means '2 bytes per float16'
                attrs.cuda.dynamic_smem_bytes = dynamic_smem_bytes

                if not transpose_b:
                    b = as_tensor_pointer(b_ptr, target_float_type.name, b_head + [k_size, n_size])
                else:
                    b = as_tensor_pointer(b_ptr, target_float_type.name, b_head + [n_size, k_size])

                # smem_storage = dyn_smem_storage
                smem_a = tensor_pointer(
                    target_float_type.name, shape=[2, block_m, block_k], layout=row_major(2) + smem_a_type.layout
                )

                if not transpose_b:
                    smem_b = tensor_pointer(
                        target_float_type.name, shape=[2, block_k, block_n], layout=row_major(2) + smem_b_type.layout
                    )
                else:
                    smem_b = tensor_pointer(
                        target_float_type.name, shape=[2, block_n, block_k], layout=row_major(2) + smem_b_type.layout
                    )

                smem_a = dynamic_shared_memory(byte_offset=0, dtype=float16)
                smem_b = dynamic_shared_memory(byte_offset=2 * block_m * block_k * 2, dtype=float16)
                regs_a = register_tensor(target_float_type, [2, mma_count_m, mma_config.a_elements])
                regs_b = register_tensor(target_float_type, [2, mma_count_n, mma_config.b_elements])
                regs_c = register_tensor(acc_dtype, [mma_count_m, mma_count_n, mma_config.c_elements])

                for i, j, p in grid(mma_count_m, mma_count_n, mma_config.c_elements):
                    regs_c[i, j, p] = 0.0

                load_smem_a(0, a, ~smem_a[0, 0, 0])
                load_smem_b(0, b, ~smem_b[0, 0, 0])
                cp_async_wait_all()

                syncthreads()
                for k0 in range((k_part_extent + block_k - 1) // block_k):
                    load_smem_a(k0 + 1, a, ~smem_a[(k0 + 1) % 2, 0, 0])
                    load_smem_b(k0 + 1, b, ~smem_b[(k0 + 1) % 2, 0, 0])
                    for mi in range(mma_count_m):
                        load_regs_a(mi, 0, ~smem_a[k0 % 2, 0, 0], ~regs_a[0, mi, 0])
                    for mj in range(mma_count_n // 2):
                        load_regs_b_2x(mj * 2, 0, ~smem_b[k0 % 2, 0, 0], ~regs_b[0, mj * 2, 0])
                    for mk in range(mma_count_k):
                        if mk + 1 < mma_count_k:
                            for mi in range(mma_count_m):
                                load_regs_a(mi, mk + 1, ~smem_a[k0 % 2, 0, 0], ~regs_a[(mk + 1) % 2, mi, 0])
                            for mj in range(mma_count_n // 2):
                                load_regs_b_2x(mj * 2, mk + 1, ~smem_b[k0 % 2, 0, 0], ~regs_b[(mk + 1) % 2, mj * 2, 0])
                        for mi, mj in grid(mma_count_m, mma_count_n):
                            warp_mma(~regs_a[mk % 2, mi, 0], ~regs_b[mk % 2, mj, 0], ~regs_c[mi, mj, 0])
                    cp_async_wait_all()
                    syncthreads()

                # store back
                warp_id, lane_id = threadIdx.x / 32, threadIdx.x % 32

                if warp_count_k == 1:
                    store_c_reg2gmem(regs_c, c)
                else:
                    smem_c = tensor_pointer(acc_dtype, shape=[block_m, block_n])
                    smem_c = dynamic_shared_memory(byte_offset=0, dtype=acc_dtype)

                    for k_round in range(warp_count_k):
                        for wi, wj, wk in spatial(warp_count_m, warp_count_n, warp_count_k).on(warp_id):
                            if wk == k_round:
                                for mi, mj in grid(mma_count_m, mma_count_n):
                                    p = 0
                                    for i, j in mma_config.c_store_map.on(lane_id):
                                        delta_m = wi * warp_m + mi * mma_m + i
                                        delta_n = wj * warp_n + mj * mma_n + j
                                        if k_round == 0:
                                            smem_c[delta_m, delta_n] = regs_c[mi, mj, p]
                                        else:
                                            smem_c[delta_m, delta_n] += regs_c[mi, mj, p]
                                        p += 1
                        if warp_count_k > 1:
                            syncthreads()
                    store_c_smem2gmem(smem_c, c)

            nonempty_c_head: bool = len(c_head_non_parallel_k) > 0

            @hidet.script
            def reduce(temp_c_buffer: target_float_type[parallel_c_shape], c: target_float_type[c_shape]):

                attrs.cuda.grid_dim = cdiv(m_size, block_m), cdiv(n_size, block_n), prod(c_head_non_parallel_k)
                attrs.cuda.block_dim = threads
                attrs.cuda.dynamic_shared_memory = 0

                bidx = blockIdx.x
                bidy = blockIdx.y
                bidz = blockIdx.z

                tensor_sum = make_tensor(target_float_type, store_c_layout, "register")
                fill(tensor_sum, 0.0)

                for i in range(parallel_k_parts):
                    partial_sum = make_tensor(target_float_type, auto_layout, "register")
                    if nonempty_c_head:
                        temp = temp_c_buffer[i, bidz, bidx * block_m :, bidy * block_n :]
                    else:
                        temp = temp_c_buffer[i, bidx * block_m :, bidy * block_n :]
                    tensor_temp = tensor_view(temp, TensorLayout((block_m, block_n), (n_size, 1)), "global")
                    txgtmp = partition_src(tensor_temp, auto_copy())
                    txrpsum = partition_dst(partial_sum, auto_copy())
                    extents = [m_size - bidx * block_m, n_size - bidy * block_n]
                    mask_tmp = mask(auto_copy(), extents)
                    copy(auto_copy((block_m, block_n)), txgtmp, txrpsum, mask_tmp)
                    tensor_sum = tensor_sum + partial_sum

                extents = [m_size - bidx * block_m, n_size - bidy * block_n]
                if parallel_k_parts > 1:
                    if nonempty_c_head:
                        offsets = [bidz, bidx * block_m, bidy * block_n]
                    else:
                        offsets = [bidx * block_m, bidy * block_n]
                    collective_store(tiled_copy, tensor_sum, c, offsets, extents)
                else:

                    if nonempty_c_head:
                        cc = c[bidz, bidx * block_m :, bidy * block_n :]
                    else:
                        cc = c[bidx * block_m :, bidy * block_n :]

                    tensor_c = tensor_view(cc, TensorLayout((block_m, block_n), (n_size, 1)), "global")
                    mask_c = mask(tiled_copy, extents)
                    txgc = partition_dst(tensor_c, tiled_copy)
                    txrsum = partition_src(tensor_sum, tiled_copy)
                    copy(tiled_copy, txrsum, txgc, mask_c)

            @hidet.script
            def launch(a: target_float_type[a_shape], b: target_float_type[b_shape], c: target_float_type[c_shape]):
                attrs.func_kind = "public"
                if parallel_k_parts > 1:
                    temp_c_buffer_void = request_cuda_workspace(
                        prod(parallel_c_shape) * float16.nbytes, require_clean=True
                    )
                    temp_c_buffer = as_tensor_pointer(temp_c_buffer_void, target_float_type, parallel_c_shape)
                    matmul_f16_kernel(a, b, temp_c_buffer)
                    reduce(temp_c_buffer, c)
                else:
                    matmul_f16_kernel(a, b, c)

        ir_module = module.ir_module()
        assert isinstance(matmul_f16_kernel, Function)

        return ir_module


class MatmulF16CuteOp(Operator):
    def __init__(self, a: Tensor, b: Tensor, acc_dtype: Union[DataType, str], transpose_b: bool = False):
        super().__init__(
            inputs=[a, b],
            attributes={'acc_dtype': acc_dtype, 'transpose_b': transpose_b},
            task=MatmulF16CuteTask(input_like(a, 'a'), input_like(b, 'b'), acc_dtype, transpose_b),
        )


def matmul_f16_cute(
    a: Tensor, b: Tensor, acc_dtype: Union[DataType, str] = "float32", transpose_b: bool = False
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

    if not a.dtype.is_any_float16() or not b.dtype.is_any_float16():
        raise ValueError('matmul_f16_cute only supports float16 or bfloat16, got {} and {}'.format(a.dtype, b.dtype))

    acc_dtype = data_type(acc_dtype)
    return MatmulF16CuteOp(a, b, acc_dtype, transpose_b).outputs[0]
