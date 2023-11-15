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
from typing import List, Union
from hidet.ir.dtypes import float32, int32
from hidet.ir.expr import cast
from hidet.ir.module import IRModule
from hidet.ir.compute import TensorNode
from hidet.ir.primitives import printf
from hidet.ir.primitives.cpu import avx_f32x8_setzero, avx_f32x8_load_aligned, avx_free, avx_malloc
from hidet.ir.stmt import DeclareScope
from hidet.ir.task import Task
from hidet.ir.compute import compute, reduce
from hidet.graph.ops.utils import input_like, broadcast_shape, can_mutually_broadcast
from hidet.ir.library import tune
from hidet.graph.operator import Operator, Tensor
from hidet.graph.ops.utils import broadcast_indices


class MatmulF32Taskx86_refactored(Task):

    def __init__(self, a: TensorNode, b: TensorNode):
        a_shape = a.const_shape
        b_shape = b.const_shape

        if not a.type.dtype == float32 or not b.type.dtype == float32:
            raise ValueError('Both inputs must be float32 tensors')

        if len(a_shape) < 2 or len(b_shape) < 2:
            raise ValueError('Matrix multiplication expect at least 2D tensor, got {} and {}'.format(a_shape, b_shape))

        self._assert(
            a_shape[-1] == b_shape[-2],
            msg=(
                'Matrix multiplication expect tensor A and B with shape [..., M, K] and [..., K, N]'
                ', got {} and {}'.format(a_shape, b_shape)
            ),
        )

        self._assert(
            can_mutually_broadcast(a_shape[:-2], b_shape[:-2]),
            msg=(
                'Matrix multiplication expect tensor A and B with compatible broadcast shape, '
                'got {} and {}'.format(a_shape, b_shape)
            ),
        )

        k_size = a_shape[-1]
        c_shape = broadcast_shape(a_shape[:-2], b_shape[:-2]) + [a_shape[-2], b_shape[-1]]

        c = compute(
            name='c',
            shape=c_shape,
            fcompute=lambda *indices: reduce(
                shape=[k_size],
                fcompute=lambda k: a[broadcast_indices(indices[:-2], a_shape[:-2], c_shape[1:-2]) + [indices[-2], k]]
                                   * b[broadcast_indices(indices[:-2], b_shape[:-2], c_shape[1:-2]) + [k, indices[-1]]],
                reduce_type='sum',
            ),
        )

        super().__init__(
            name='matmul_f32_x86_v2',
            inputs=[a, b],
            outputs=[c],
            attributes={'m_size': a_shape[-2], 'n_size': b_shape[-1], 'k_size': a_shape[-1]},
        )

    def allow_epilogue(self) -> bool:
        return True

    def allow_prologue(self) -> bool:
        return False

    def implement_cpu(self, working_dir: str) -> Union[IRModule, List[IRModule]]:
        return tune.extract_ir_modules(self.schedule_matmulf32_x86)

    @tune.space(1, MC=[2016], NC=[256, 384, 512], KC=[384, 512, 560], ways=[(1, 1, 1, 1)])
    def schedule_matmulf32_x86(
            self, MC=2016, NC=256, KC=560, ways=(1, 8, 2, 1)
    ) -> IRModule:
        import hidet
        from hidet.ir.type import tensor_type
        from hidet.lang import tensor, grid, as_tensor_pointer
        from hidet.lang.layout import row_major, column_major
        from hidet.lang.cpu import avx_f32x8_store, avx_f32x8_fmadd, avx_f32x8_load, avx_f32x8_broadcast
        from hidet.lang.cpu import avx_f32x8_store_aligned, avx_f32x8_load_aligned
        from hidet.lang.cpu import avx_f32x8_unpacklo, avx_f32x8_unpackhi
        from hidet.lang.cpu import avx_f32x8_shuffle, avx_f32x8_cast_f32x4
        from hidet.lang.cpu import avx_f32x8_insert_f32x4, avx_f32x8_permute2f32x4
        from hidet.lang.cpu import cpu_atomic_load_n, cpu_atomic_add_fetch, cpu_atomic_fetch_xor

        node_a, node_b = self.inputs[0], self.inputs[1]
        a_shape = node_a.const_shape
        b_shape = node_b.const_shape
        m_size, n_size, k_size = a_shape[-2], b_shape[-1], a_shape[-1]

        MR, NR = 6, 16

        tune.check(MC % MR == NC % NR == 0, 'Tile size must divide the corresponding block size')

        with hidet.script_module() as module:
            # Get the number of threads...
            loop5_nways, loop3_nways, macro_nways, loop1_nways = ways
            loop4_nways = 1
            nthreads = loop5_nways * loop3_nways * macro_nways * loop1_nways

            packa_thrcomm_barrier_sense = module.define_global_var(
                name="pack_a_barrier_sense",
                var_type=int32[nthreads]
            )
            packa_thrcomm_threads_arrived = module.define_global_var(
                name="pack_a_threads_arrived",
                var_type=int32[nthreads]
            )

            packb_thrcomm_barrier_sense = module.define_global_var(
                name='pack_b_barrier_sense',
                var_type=int32[nthreads]
            )
            packb_thrcomm_barrier_threads_arrived = module.define_global_var(
                name="pack_b_threads_arrived",
                var_type=int32[nthreads]
            )

            @hidet.script
            def init_thr(sense: ~int32, arrived: ~int32, size: int32):
                for i in range(size):
                    sense[i] = 0
                    arrived[i] = 0

            init_thr.kind = "cpu_internal"

            # Helpers
            packed_a_type = tensor_type('float32', layout=row_major(MC // MR,
                                                                    1) * column_major(
                MR, KC))
            packed_b_type = tensor_type('float32', layout=row_major(1,
                                                                    NC // NR) * row_major(
                KC, NR))

            # Get the number of threads remaining at each level
            loop5_nthreads = nthreads
            loop4_nthreads = nthreads // loop5_nways
            loop3_nthreads = loop4_nthreads
            macro_nthreads = loop3_nthreads // loop3_nways
            loop1_nthreads = macro_nthreads // macro_nways

            packb_nthreads = loop3_nthreads
            packa_nthreads = macro_nthreads

            packed_a_buffers_needed = loop3_nways * loop5_nways

            @hidet.script
            def thread_range_sub(n_way: int32, work_id: int32, n: int32, bf: int32, start: ~int32, end: ~int32):
                if n_way == 1:
                    start[0] = 0
                    end[0] = n
                    return
                all_start = 0
                all_end = n
                size = all_end - all_start

                n_bf_whole = size // bf
                n_bf_left = size % bf

                n_bf_lo = n_bf_whole // n_way
                n_bf_hi = n_bf_whole // n_way

                n_th_lo = n_bf_whole % n_way
                # If some partitions must have more block_factors than others, assign the slightly larger partitions to lower index threads
                if n_th_lo != 0:
                    n_bf_lo += 1
                # Compute the actual widths (in units of rows/columns) of individual threads in the low and high groups
                size_lo = n_bf_lo * bf
                size_hi = n_bf_hi * bf

                # Pre-compute the starting indices of the low and high groups
                lo_start = all_start
                hi_start = all_start + n_th_lo * size_lo

                # Compute the start and end of individual threads' ranges
                if work_id < n_th_lo:
                    start[0] = lo_start + work_id * size_lo
                    end[0] = lo_start + (work_id + 1) * size_lo
                else:
                    start[0] = hi_start + (work_id - n_th_lo) * size_hi
                    end[0] = hi_start + (work_id - n_th_lo + 1) * size_hi

                    # Add the remainder to the last thread's end
                    if work_id == n_way - 1:
                        end[0] += n_bf_left
                        end[0] = min(end[0], all_end)

            thread_range_sub.kind = "cpu_internal"

            @hidet.script
            def thread_range_jrir(work_id: int32, n_way: int32, n: int32, bf: int32,
                                  start: ~int32, end: ~int32, inc: ~int32):
                start[0] = work_id
                end[0] = n
                inc[0] = n_way

            thread_range_jrir.kind = "cpu_internal"

            @hidet.script
            def determine_blocksize_f_sub(i: int32, dim: int32, b_alg: int32) -> int32:
                dim_left_now = dim - i
                b_now = -1
                if dim_left_now <= b_alg:
                    b_now = dim_left_now
                else:
                    b_now = b_alg
                assert b_now >= 0
                return b_now

            determine_blocksize_f_sub.kind = "cpu_internal"

            @hidet.script
            def not_edge(i: int32, n_iter: int32, n_left: int32) -> bool:
                return i != n_iter - 1 or n_left == 0

            not_edge.kind = 'cpu_internal'

            # TODO: Is this the way to find out the "index" of the packed A buffer?
            @hidet.script
            def packa_index(work_id_loop5: int32, work_id_loop3: int32) -> int32:
                return work_id_loop5 * loop3_nways + work_id_loop3

            packa_index.kind = 'cpu_internal'

            # Thread barrier
            @hidet.script
            def thrcomm_barrier(tid: int32, barrier_sense: ~int32,
                                barrier_threads_arrived: ~int32, num_threads: int32):
                if num_threads == 1:
                    return
                orig_sense = cpu_atomic_load_n(barrier_sense, 0)  # _ATOMIC_RELAXED

                # Register the current thread's arrival by incrementing the counter
                my_threads_arrived = cpu_atomic_add_fetch(
                    barrier_threads_arrived, 1, 4)  # _ATOMIC_ACQ_REL

                if my_threads_arrived == num_threads:
                    barrier_threads_arrived[0] = 0
                    cpu_atomic_fetch_xor(barrier_sense, 1, 3)  # _ATOMIC_RELEASE
                else:
                    while cpu_atomic_load_n(barrier_sense, 2) == orig_sense:  # _ATOMIC_ACQUIRE
                        pass

            thrcomm_barrier.kind = 'cpu_internal'

            @hidet.script
            def micro_kernel(
                    a: packed_a_type, b: packed_b_type, c_ptr: ~float32, pb: int32, msize: int32, nsize: int32,
                    is_first: bool
            ):
                c = as_tensor_pointer(c_ptr, dtype=float32, shape=[msize, nsize])
                c0 = avx_f32x8_load(~c[0, 0])
                c08 = avx_f32x8_load(~c[0, 8])
                c1 = avx_f32x8_load(~c[1, 0])
                c18 = avx_f32x8_load(~c[1, 8])
                c2 = avx_f32x8_load(~c[2, 0])
                c28 = avx_f32x8_load(~c[2, 8])
                c3 = avx_f32x8_load(~c[3, 0])
                c38 = avx_f32x8_load(~c[3, 8])
                c4 = avx_f32x8_load(~c[4, 0])
                c48 = avx_f32x8_load(~c[4, 8])
                c5 = avx_f32x8_load(~c[5, 0])
                c58 = avx_f32x8_load(~c[5, 8])

                if is_first:
                    c0 = avx_f32x8_setzero()
                    c08 = avx_f32x8_setzero()
                    c1 = avx_f32x8_setzero()
                    c18 = avx_f32x8_setzero()
                    c2 = avx_f32x8_setzero()
                    c28 = avx_f32x8_setzero()
                    c3 = avx_f32x8_setzero()
                    c38 = avx_f32x8_setzero()
                    c4 = avx_f32x8_setzero()
                    c48 = avx_f32x8_setzero()
                    c5 = avx_f32x8_setzero()
                    c58 = avx_f32x8_setzero()
                a_ptr = cast(a, ~float32)
                b_ptr = cast(b, ~float32)

                # TODO: For now, let's forget about unrolling for now.
                for _ in range(pb):
                    bb0to7 = avx_f32x8_load_aligned(b_ptr)
                    bb8to15 = avx_f32x8_load_aligned(b_ptr + 8)

                    aa1 = avx_f32x8_broadcast(a_ptr)
                    c0 = avx_f32x8_fmadd(aa1, bb0to7, c0)
                    c08 = avx_f32x8_fmadd(aa1, bb8to15, c08)

                    aa2 = avx_f32x8_broadcast(a_ptr + 1)
                    c1 = avx_f32x8_fmadd(aa2, bb0to7, c1)
                    c18 = avx_f32x8_fmadd(aa2, bb8to15, c18)

                    aa3 = avx_f32x8_broadcast(a_ptr + 2)
                    c2 = avx_f32x8_fmadd(aa3, bb0to7, c2)
                    c28 = avx_f32x8_fmadd(aa3, bb8to15, c28)

                    aa4 = avx_f32x8_broadcast(a_ptr + 3)
                    c3 = avx_f32x8_fmadd(aa4, bb0to7, c3)
                    c38 = avx_f32x8_fmadd(aa4, bb8to15, c38)

                    aa5 = avx_f32x8_broadcast(a_ptr + 4)
                    c4 = avx_f32x8_fmadd(aa5, bb0to7, c4)
                    c48 = avx_f32x8_fmadd(aa5, bb8to15, c48)

                    aa6 = avx_f32x8_broadcast(a_ptr + 5)
                    c5 = avx_f32x8_fmadd(aa6, bb0to7, c5)
                    c58 = avx_f32x8_fmadd(aa6, bb8to15, c58)
                    # printf(
                    #     "List of all the aa's broadcasted in this iteration: %lf %lf %lf %lf %lf %lf\n, bb0 to to bb15: %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf\n\n",
                    #     a_ptr[0], a_ptr[1], a_ptr[2], a_ptr[3], a_ptr[4], a_ptr[5], b_ptr[0], b_ptr[1], b_ptr[2],
                    #     b_ptr[3], b_ptr[4], b_ptr[5], b_ptr[6], b_ptr[7], b_ptr[8], b_ptr[9], b_ptr[10], b_ptr[11],
                    #     b_ptr[12], b_ptr[13], b_ptr[14], b_ptr[15])

                    a_ptr = a_ptr + 6
                    b_ptr = b_ptr + 16

                # Store the results
                avx_f32x8_store(c_ptr, c0)
                avx_f32x8_store(c_ptr + 8, c08)

                avx_f32x8_store(c_ptr + nsize, c1)
                avx_f32x8_store(c_ptr + (nsize + 8), c18)

                avx_f32x8_store(c_ptr + 2 * nsize, c2)
                avx_f32x8_store(c_ptr + (2 * nsize + 8), c28)

                avx_f32x8_store(c_ptr + 3 * nsize, c3)
                avx_f32x8_store(c_ptr + (3 * nsize + 8), c38)

                avx_f32x8_store(c_ptr + 4 * nsize, c4)
                avx_f32x8_store(c_ptr + (4 * nsize + 8), c48)

                avx_f32x8_store(c_ptr + 5 * nsize, c5)
                avx_f32x8_store(c_ptr + (5 * nsize + 8), c58)

            #### Some setup code ####
            packed_b_height = KC
            if packed_b_height > k_size:
                packed_b_height = k_size
            packed_b_width = NC
            if packed_b_width > n_size:
                packed_b_width = (n_size + NR - 1) // NR * NR

            packed_b_total_width = packed_b_width * loop5_nways
            packed_b_total_size = packed_b_total_width * packed_b_height
            packed_b_individual_size = packed_b_width * packed_b_height

            packed_a_individual_height = MC
            if packed_a_individual_height > m_size:
                packed_a_individual_height = (m_size + MR - 1) // MR * MR
            # packed_a_total_height = packed_a_individual_height * loop3_nways
            # packed_a_total_height = packed_a_individual_height * loop5_nways
            packed_a_total_height = packed_a_individual_height * packed_a_buffers_needed

            packed_a_width = KC
            if packed_a_width > k_size:
                packed_a_width = k_size
            packed_a_total_size = packed_a_total_height * packed_a_width
            packed_a_individual_size = packed_a_width * packed_a_individual_height

            packb_buf_ptr = module.define_global_var(
                name='packb_buf_ptr',
                var_type=float32[packed_b_total_size]
            )
            packa_buf_ptr = module.define_global_var(
                name='packa_buf_ptr',
                var_type=float32[packed_a_total_size]
            )

            packb_buf = cast(packb_buf_ptr, ~float32)
            packa_buf = cast(packa_buf_ptr, ~float32)

            packed_a_type = tensor_type(
                dtype='float32',
                layout=row_major(packed_a_individual_height // MR, 1) * column_major(MR, packed_a_width)
            )

            ##### Start of the loops around micro kernel #####

            @hidet.script
            def gemm_pack_a(
                    loop3_partition_a: ~float32,
                    loop3_partition_a_width: int32,
                    loop3_partition_a_height: int32,
                    packed_a_buf: ~float32,
                    comm_id_packa: int32,
                    work_id_packa: int32,
                    packa_nways: int32
            ):
                packed_a_tensor = as_tensor_pointer(
                    packed_a_buf,
                    float32,
                    layout=row_major(packed_a_individual_height // MR, 1) *
                           column_major(MR, packed_a_width)
                )

                npanels_full_a = loop3_partition_a_height // MR
                panel_a_remainder = loop3_partition_a_height % MR

                npanels_a = npanels_full_a + (1 if panel_a_remainder > 0 else 0)

                for ii_panel in range(npanels_a):
                    if ii_panel % packa_nthreads != work_id_packa % packa_nthreads:
                        continue

                    a_curr_panel_row_start = ii_panel * MR
                    a_curr_panel_height = min(MR,
                                              loop3_partition_a_height - a_curr_panel_row_start)

                    if a_curr_panel_height == MR:  # unroll the packing by 8
                        k_iters = loop3_partition_a_width // 8
                        k_remainder = loop3_partition_a_width % 8
                        col = 0
                        for k_iter in range(k_iters):
                            col = k_iter * 8
                            a_curr_panel_col = loop3_partition_a + (
                                    a_curr_panel_row_start * k_size + col
                            )

                            v0 = avx_f32x8_load(a_curr_panel_col)
                            v1 = avx_f32x8_load(a_curr_panel_col + k_size)
                            v2 = avx_f32x8_load(a_curr_panel_col + (2 * k_size))
                            v3 = avx_f32x8_load(a_curr_panel_col + (3 * k_size))
                            v4 = avx_f32x8_load(a_curr_panel_col + (4 * k_size))
                            v5 = avx_f32x8_load(a_curr_panel_col + (5 * k_size))

                            unpack0 = avx_f32x8_unpacklo(v0, v1)
                            unpack1 = avx_f32x8_unpackhi(v0, v1)
                            unpack2 = avx_f32x8_unpacklo(v2, v3)
                            unpack3 = avx_f32x8_unpackhi(v2, v3)
                            unpack4 = avx_f32x8_unpacklo(v4, v5)
                            unpack5 = avx_f32x8_unpackhi(v4, v5)

                            shf0 = avx_f32x8_shuffle(unpack0, unpack2, 0x44)
                            shf1 = avx_f32x8_shuffle(unpack4, unpack0, 0xE4)
                            shf2 = avx_f32x8_shuffle(unpack2, unpack4, 0xEE)
                            shf3 = avx_f32x8_shuffle(unpack5, unpack1, 0xE4)
                            shf4 = avx_f32x8_shuffle(unpack3, unpack5, 0xEE)
                            shf5 = avx_f32x8_shuffle(unpack1, unpack3, 0x44)

                            low_shf1 = avx_f32x8_cast_f32x4(shf1)
                            res0 = avx_f32x8_insert_f32x4(shf0, low_shf1, 0x1)
                            res1 = avx_f32x8_permute2f32x4(shf0, shf1, 0x31)

                            low_shf5 = avx_f32x8_cast_f32x4(shf5)
                            res2 = avx_f32x8_insert_f32x4(shf2, low_shf5, 0x1)
                            res3 = avx_f32x8_permute2f32x4(shf2, shf5, 0x31)

                            low_shf4 = avx_f32x8_cast_f32x4(shf4)
                            res4 = avx_f32x8_insert_f32x4(shf3, low_shf4, 0x1)
                            res5 = avx_f32x8_permute2f32x4(shf3, shf4, 0x31)

                            # TODO: Now I changed to unaligned to debug...
                            avx_f32x8_store(
                                ~packed_a_tensor[a_curr_panel_row_start, col],
                                res0
                            )
                            avx_f32x8_store(
                                ~packed_a_tensor[a_curr_panel_row_start + 2,
                                                 col + 1],
                                res2
                            )
                            avx_f32x8_store(
                                ~packed_a_tensor[a_curr_panel_row_start + 4,
                                                 col + 2],
                                res4)
                            avx_f32x8_store(
                                ~packed_a_tensor[a_curr_panel_row_start,
                                col + 4],
                                res1
                            )
                            avx_f32x8_store(
                                ~packed_a_tensor[a_curr_panel_row_start + 2,
                                                 col + 5],
                                res3
                            )
                            avx_f32x8_store(
                                ~packed_a_tensor[a_curr_panel_row_start + 4,
                                                 col + 6],
                                res5
                            )
                        remaining_start_col = k_iters * 8
                        for remain_off in range(k_remainder):
                            curr_remain_col = remaining_start_col + remain_off
                            for micropanel_row in range(MR):
                                packed_a_tensor[
                                    a_curr_panel_row_start + micropanel_row,
                                    curr_remain_col] = \
                                    loop3_partition_a[(
                                                              micropanel_row + a_curr_panel_row_start) * k_size + curr_remain_col]
                    else:
                        remain_start_row = npanels_full_a * MR
                        for remain_col in range(loop3_partition_a_width):
                            for remain_row in range(panel_a_remainder):
                                packed_a_tensor[
                                    remain_start_row + remain_row, remain_col] = \
                                    loop3_partition_a[(
                                                              remain_row + remain_start_row) * k_size + remain_col]
                            remain_row = panel_a_remainder
                            while remain_row < MR:
                                packed_a_tensor[
                                    remain_start_row + remain_row, remain_col] = 0.0
                                remain_row += 1

            @hidet.script
            def gemm_pack_b(
                    loop4_partition_b: ~float32,
                    loop4_partition_b_width: int32,
                    loop4_partition_b_height: int32,
                    packed_b_buf: ~float32,
                    comm_id_packb: int32, work_id_packb: int32,
                    packb_nways: int32
            ):
                npanels_full_b = loop4_partition_b_width // NR
                npanels_b_remainder = loop4_partition_b_width % NR

                npanels_b = npanels_full_b + (1 if npanels_b_remainder != 0 else 0)
                packedb_panel_stride = packed_b_height * NR

                # Loop for the packing of B
                for i_panel in range(npanels_b):
                    if i_panel % packb_nthreads != work_id_packb % packb_nthreads:
                        continue
                    packed_b_buff_curr = packed_b_buf + (
                            i_panel * packedb_panel_stride)
                    curr_panel_start = i_panel * NR
                    curr_panel_width = min(NR,
                                           loop4_partition_b_width - curr_panel_start)

                    if curr_panel_width == NR:
                        k_iters = loop4_partition_b_height // 8
                        k_remainder = loop4_partition_b_height % 8

                        row = 0
                        for k_iter in range(k_iters):
                            row = k_iter * 8
                            b_panel = loop4_partition_b + (
                                    row * n_size + curr_panel_start)
                            b00 = avx_f32x8_load(b_panel)
                            b08 = avx_f32x8_load(b_panel + 8)

                            avx_f32x8_store(packed_b_buff_curr, b00)
                            avx_f32x8_store(packed_b_buff_curr + 8, b08)
                            packed_b_buff_curr += 16

                            b10 = avx_f32x8_load(b_panel + n_size)
                            b18 = avx_f32x8_load(b_panel + (n_size + 8))

                            avx_f32x8_store(packed_b_buff_curr, b10)
                            avx_f32x8_store(packed_b_buff_curr + 8, b18)
                            packed_b_buff_curr += 16

                            b20 = avx_f32x8_load(b_panel + (2 * n_size))
                            b28 = avx_f32x8_load(b_panel + (2 * n_size + 8))

                            avx_f32x8_store(packed_b_buff_curr, b20)
                            avx_f32x8_store(packed_b_buff_curr + 8, b28)
                            packed_b_buff_curr += 16

                            b30 = avx_f32x8_load(b_panel + (3 * n_size))
                            b38 = avx_f32x8_load(b_panel + (3 * n_size + 8))

                            avx_f32x8_store(packed_b_buff_curr, b30)
                            avx_f32x8_store(packed_b_buff_curr + 8, b38)
                            packed_b_buff_curr += 16

                            b40 = avx_f32x8_load(b_panel + (4 * n_size))
                            b48 = avx_f32x8_load(b_panel + (4 * n_size + 8))

                            avx_f32x8_store(packed_b_buff_curr, b40)
                            avx_f32x8_store(packed_b_buff_curr + 8, b48)
                            packed_b_buff_curr += 16

                            b50 = avx_f32x8_load(b_panel + (5 * n_size))
                            b58 = avx_f32x8_load(b_panel + (5 * n_size + 8))

                            avx_f32x8_store(packed_b_buff_curr, b50)
                            avx_f32x8_store(packed_b_buff_curr + 8, b58)
                            packed_b_buff_curr += 16

                            b60 = avx_f32x8_load(b_panel + (6 * n_size))
                            b68 = avx_f32x8_load(b_panel + (6 * n_size + 8))

                            avx_f32x8_store(packed_b_buff_curr, b60)
                            avx_f32x8_store(packed_b_buff_curr + 8, b68)
                            packed_b_buff_curr += 16

                            b70 = avx_f32x8_load(b_panel + (7 * n_size))
                            b78 = avx_f32x8_load(b_panel + (7 * n_size + 8))

                            avx_f32x8_store(packed_b_buff_curr, b70)
                            avx_f32x8_store(packed_b_buff_curr + 8, b78)

                            packed_b_buff_curr += 16

                        row = k_iters * 8
                        for _ in range(k_remainder):
                            b_panel = loop4_partition_b + (
                                    row * n_size + curr_panel_start)
                            b00 = avx_f32x8_load(b_panel)
                            b08 = avx_f32x8_load(b_panel + 8)
                            avx_f32x8_store(packed_b_buff_curr, b00)
                            avx_f32x8_store(packed_b_buff_curr + 8, b08)
                            packed_b_buff_curr += 16
                            row += 1

                    else:
                        packed_b_remaining_buf = packed_b_buf + (
                                npanels_full_b * packedb_panel_stride)
                        if npanels_b_remainder > 0:
                            # TODO: I think this if should always be true if this is executed?
                            remain_col_start = npanels_full_b * NR
                            for remain_row in range(loop4_partition_b_height):
                                packed_b_remaining_buf_curr = packed_b_remaining_buf + (
                                        remain_row * NR)
                                for remain_col in range(npanels_b_remainder):
                                    packed_b_remaining_buf_curr[0] = \
                                        loop4_partition_b[
                                            (remain_row * n_size) + (
                                                    remain_col_start + remain_col)
                                            ]
                                    packed_b_remaining_buf_curr += 1
                                zero_fill_col = npanels_b_remainder
                                while zero_fill_col < NR:
                                    packed_b_remaining_buf_curr[0] = 0.0
                                    packed_b_remaining_buf_curr += 1
                                    zero_fill_col += 1

            gemm_pack_b.kind = "cpu_internal"
            gemm_pack_a.kind = "cpu_internal"
            micro_kernel.kind = "cpu_internal"

            @hidet.script
            def gemm_macro(
                    packed_a: ~float32,
                    packed_b: ~float32,
                    c: float32[m_size, n_size],
                    c_row_off: int32,
                    c_col_off: int32,
                    macro_m: int32,
                    macro_n: int32,
                    macro_k: int32,
                    ps_packed_a: int32,
                    ps_packed_b: int32,
                    comm_id_macro: int32,
                    work_id_macro: int32,
                    is_first: bool, work_id_3rd_loop: int32, work_id_4th_loop: int32, work_id_5th_loop: int32
            ):
                # assert loop1_nthreads == 1
                comm_id_1st_loop = comm_id_macro % loop1_nthreads
                work_id_1st_loop = comm_id_1st_loop // (loop1_nthreads // loop1_nways)

                n_iter = macro_n // NR
                n_remainder = macro_n % NR
                m_iter = macro_m // MR
                m_remainder = macro_m % MR

                if n_remainder > 0:
                    n_iter += 1
                if m_remainder > 0:
                    m_iter += 1

                jr_start = -1
                jr_end = -1
                ir_start = -1
                ir_end = -1
                jr_inc = -1
                ir_inc = -1

                thread_range_jrir(
                    work_id_macro,
                    macro_nways,
                    n_iter,
                    1,
                    ~jr_start,
                    ~jr_end,
                    ~jr_inc
                )

                thread_range_jrir(
                    work_id_1st_loop,
                    loop1_nways,
                    m_iter,
                    1,
                    ~ir_start,
                    ~ir_end,
                    ~ir_inc
                )

                rs_packeda = 1
                rstep_a = ps_packed_a
                cstep_b = ps_packed_b

                cstep_c = NR
                rstep_c = n_size * MR

                macro_c_cast = as_tensor_pointer(
                    ~c[c_row_off, c_col_off],
                    dtype=float32,
                    shape=(m_size, n_size)
                )
                temp_c = tensor(scope=DeclareScope.Default,
                                dtype=float32,
                                layout=row_major(MR, NR),
                                is_static=False)
                j = jr_start
                while j < jr_end:
                    b1 = packed_b + j * cstep_b
                    c1 = macro_c_cast + j * cstep_c
                    n_cur = NR if not_edge(j, n_iter, n_remainder) else n_remainder

                    i = ir_start
                    while i < ir_end:
                        a1 = packed_a + i * rstep_a
                        c11 = c1 + i * rstep_c
                        c11 = as_tensor_pointer(c11, dtype=float32, shape=(m_size, n_size))
                        m_cur = MR if not_edge(i, m_iter, m_remainder) else m_remainder
                        #
                        # printf(''' work_id_macro: %d, work_id_3rd_loop: %d, work_id_4th_loop: %d, work_id_5th_loop: %d, c_row_off: %d, c_col_off: %d,
                        #        macro_m: %d, macro_n: %d, macro_k: %d,
                        #        ps_packed_a: %d, ps_packed_b: %d, ,
                        #        n_iter: %d, n_remainder: %d, m_iter: %d, m_remainder: %d,
                        #        jr_start: %d, jr_end: %d, jr_inc: %d,
                        #        ir_start: %d, ir_end: %d, ir_inc: %d,
                        #        i: %d, j: %d,
                        #        rstep_a: %d, cstep_b: %d, cstep_c: %d, rstep_c: %d,
                        #        n_cur: %d, m_cur: %d \n\n''',
                        #        work_id_macro, work_id_3rd_loop, work_id_4th_loop, work_id_5th_loop, c_row_off, c_col_off,
                        #        macro_m, macro_n, macro_k,
                        #        ps_packed_a, ps_packed_b,
                        #        n_iter, n_remainder, m_iter, m_remainder,
                        #        jr_start, jr_end, jr_inc,
                        #        ir_start, ir_end, ir_inc,
                        #        i, j,
                        #        rstep_a, cstep_b, cstep_c, rstep_c,
                        #        n_cur, m_cur)

                        if m_cur == MR and n_cur == NR:
                            # micro_kernel(a1, b1, c11, macro_k, macro_m, macro_n, is_first)
                            micro_kernel(a1, b1, c11, macro_k, m_size, n_size, is_first)
                        else:
                            for i, j in grid(MR, NR):
                                temp_c[i, j] = 0.0
                            # micro_kernel(a1, b1, cast(temp_c, ~float32), macro_k, MR, NR, is_first)
                            micro_kernel(a1, b1, cast(temp_c, ~float32), macro_k, MR, NR, is_first)
                            if not is_first:
                                for mm, nn in grid(m_cur, n_cur):
                                    c11[mm, nn] += temp_c[mm, nn]
                            else:
                                for mm, nn in grid(m_cur, n_cur):
                                    c11[mm, nn] = temp_c[mm, nn]
                                    # c11[mm, nn] += temp_c[mm, nn]

                        i += ir_inc
                    j += jr_inc

            gemm_macro.kind = "cpu_internal"

            @hidet.script
            def gemm_3rd_loop(
                    a: float32[m_size, k_size],
                    packed_b: ~float32,
                    c: float32[m_size, n_size],
                    loop3_partition_a_start_col: int32,
                    loop3_partition_b_start_col: int32,
                    loop3_partition_a_width: int32,
                    loop3_partition_b_width: int32,
                    comm_id_3rd_loop: int32,
                    work_id_3rd_loop: int32,
                    is_first: bool, work_id_4th_loop: int32, work_id_5th_loop: int32):
                comm_id_macro = comm_id_3rd_loop % macro_nthreads
                work_id_macro = comm_id_macro // (macro_nthreads // macro_nways)
                comm_id_packa = comm_id_macro
                work_id_packa = comm_id_macro
                packa_nways = macro_nthreads

                m_start_loop3 = 0
                m_end_loop3 = 0
                thread_range_sub(
                    loop3_nways,
                    work_id_3rd_loop,
                    m_size,
                    MR,
                    ~m_start_loop3,
                    ~m_end_loop3
                )

                ii = m_start_loop3
                while ii < m_end_loop3:
                    b_alg_loop3 = determine_blocksize_f_sub(
                        ii, m_size, MC
                    )
                    b_alg_loop3 = min(b_alg_loop3, m_end_loop3 - ii)
                    loop3_partition_a_start_row = ii
                    loop3_partition_a_height = b_alg_loop3

                    loop3_partition_a = cast(a, ~float32) + (
                            loop3_partition_a_start_row * k_size +
                            loop3_partition_a_start_col
                    )

                    # packed_a_buf = packa_buf + (work_id_3rd_loop * packed_a_individual_size)
                    # packed_a_buf = packa_buf + (work_id_5th_loop * packed_a_individual_size)
                    packed_a_idx = packa_index(work_id_5th_loop, work_id_3rd_loop)
                    packed_a_buf = packa_buf + (packed_a_idx * packed_a_individual_size)

                    # printf(
                    #     "work_id_3rd_loop: %d, work_id_4th_loop: %d, work_id_5th_loop: %d, "
                    #     "loop3_partition_a_start_col: %d, loop3_partition_b_start_col: %d, "
                    #     "loop3_partition_a_width: %d, loop3_partition_b_width: %d, "
                    #     "loop3_partition_a_start_row: %d, loop3_partition_a_height: %d, "
                    #     "m_start_loop3: %d, m_end_loop3: %d, ii: %d, b_alg_loop3: %d, packed_a_idx: %d\n\n",
                    #     work_id_3rd_loop, work_id_4th_loop, work_id_5th_loop,
                    #     loop3_partition_a_start_col, loop3_partition_b_start_col,
                    #     loop3_partition_a_width, loop3_partition_b_width,
                    #     loop3_partition_a_start_row, loop3_partition_a_height,
                    #     m_start_loop3, m_end_loop3, ii, b_alg_loop3, packed_a_idx)

                    # TODO: If passed, see if this barrier is necessary
                    # printf(
                    #     "Begin: calling the first barrier for the 3rd loop; work_id_3rd_loop: %d, comm_id_3rd_loop: %d, work_id_4th_loop: %d, work_id_5th_loop: %d, ii: %d, loop3_partition_a_start_col: %d, loop3_partition_b_start_col: %d, packed_a_idx: %d\n",
                    #     work_id_3rd_loop, comm_id_3rd_loop, work_id_4th_loop, work_id_5th_loop, ii,
                    #     loop3_partition_a_start_col,
                    #     loop3_partition_b_start_col, packed_a_idx)
                    thrcomm_barrier(
                        comm_id_packa,
                        ~packa_thrcomm_barrier_sense[packed_a_idx],
                        ~packa_thrcomm_threads_arrived[packed_a_idx],
                        packa_nthreads
                    )
                    # printf(
                    #     "End: calling the first barrier for the 3rd loop; work_id_3rd_loop: %d, comm_id_3rd_loop: %d, work_id_4th_loop: %d, work_id_5th_loop: %d, ii: %d, loop3_partition_a_start_col: %d, loop3_partition_b_start_col: %d, packed_a_idx: %d\n",
                    #     work_id_3rd_loop, comm_id_3rd_loop, work_id_4th_loop, work_id_5th_loop, ii,
                    #     loop3_partition_a_start_col,
                    #     loop3_partition_b_start_col, packed_a_idx)

                    gemm_pack_a(
                        loop3_partition_a,
                        loop3_partition_a_width,
                        loop3_partition_a_height,
                        packed_a_buf,
                        comm_id_packa,
                        work_id_packa,
                        packa_nways
                    )

                    # This marks the end of the packing of A,
                    # so a barrier is needed
                    # printf(
                    #     "Begin: calling the second barrier for the 3rd loop; work_id_3rd_loop: %d, comm_id_3rd_loop: %d, work_id_4th_loop: %d, work_id_5th_loop: %d, ii: %d, loop3_partition_a_start_col: %d, loop3_partition_b_start_col: %d\n",
                    #     work_id_3rd_loop, comm_id_3rd_loop, work_id_4th_loop, work_id_5th_loop, ii,
                    #     loop3_partition_a_start_col,
                    #     loop3_partition_b_start_col)
                    thrcomm_barrier(
                        comm_id_packa,
                        ~packa_thrcomm_barrier_sense[packed_a_idx],
                        ~packa_thrcomm_threads_arrived[packed_a_idx],
                        packa_nthreads
                    )
                    # printf(
                    #     "End: calling the second barrier for the 3rd loop; work_id_3rd_loop: %d, comm_id_3rd_loop: %d, work_id_4th_loop: %d, work_id_5th_loop: %d, ii: %d, loop3_partition_a_start_col: %d, loop3_partition_b_start_col: %d\n",
                    #     work_id_3rd_loop, comm_id_3rd_loop, work_id_4th_loop, work_id_5th_loop, ii,
                    #     loop3_partition_a_start_col,
                    #     loop3_partition_b_start_col)

                    gemm_macro(packed_a_buf,
                               packed_b,
                               c,
                               loop3_partition_a_start_row,
                               loop3_partition_b_start_col,
                               loop3_partition_a_height,
                               loop3_partition_b_width,
                               loop3_partition_a_width,
                               MR * packed_a_width,
                               packed_b_height * NR,
                               comm_id_macro,
                               work_id_macro,
                               is_first,
                               work_id_3rd_loop,
                               work_id_4th_loop,
                               work_id_5th_loop,
                               )
                    ii += b_alg_loop3

            gemm_3rd_loop.kind = "cpu_internal"

            @hidet.script
            def gemm_4th_loop(a: float32[m_size, k_size],
                              b: float32[k_size, n_size],
                              c: float32[k_size, n_size],
                              loop5_partition_b_width: int32,
                              loop5_partition_b_start_col: int32,
                              comm_id_4th_loop: int32,
                              work_id_4th_loop: int32,
                              work_id_5th_loop: int32):
                b_alg_loop4 = KC
                i_loop4 = 0

                comm_id_3rd_loop = comm_id_4th_loop % loop3_nthreads
                work_id_3rd_loop = comm_id_3rd_loop // (loop3_nthreads // loop3_nways)
                comm_id_packb = comm_id_3rd_loop
                work_id_packb = comm_id_3rd_loop

                while i_loop4 < k_size:
                    b_alg_loop4 = determine_blocksize_f_sub(i_loop4, k_size, KC)
                    b_alg_loop4 = min(b_alg_loop4, k_size - i_loop4)

                    loop4_partition_b_height = b_alg_loop4
                    loop4_partition_b_width = loop5_partition_b_width
                    loop4_partition_b_start_row = i_loop4
                    loop4_partition_b_start_col = loop5_partition_b_start_col

                    loop4_partition_a_start_col = i_loop4
                    is_first = (i_loop4 == 0)

                    # printf(
                    #     "work_id_5th_loop: %d, work_id_4th_loop: %d, i_loop4: %d, b_alg_loop4: %d, loop4_partition_b_height: %d, loop4_partition_b_width: %d,"
                    #     "loop4_partition_b_start_row: %d, loop4_partition_b_start_col: %d, loop4_partition_a_start_col: %d, is_first: %d\n",
                    #     work_id_5th_loop, work_id_4th_loop, i_loop4, b_alg_loop4, loop4_partition_b_height, loop4_partition_b_width,
                    #     loop4_partition_b_start_row,
                    #     loop4_partition_b_start_col, loop4_partition_a_start_col, is_first)

                    packed_b_buf = packb_buf + (
                            packed_b_individual_size * work_id_5th_loop
                    )

                    loop4_partition_b = cast(b, ~float32) + \
                                        (loop4_partition_b_start_row * n_size +
                                         loop4_partition_b_start_col)

                    # # # TODO: If passed, see if this barrier is really needed
                    # printf("Begin: calling the first barrier for the 4th loop; work_id_4th_loop: %d, comm_id_4th_loop: %d\n", work_id_4th_loop, comm_id_4th_loop)
                    thrcomm_barrier(
                        comm_id_packb,
                        ~packb_thrcomm_barrier_sense[work_id_5th_loop],
                        ~packb_thrcomm_barrier_threads_arrived[work_id_5th_loop],
                        packb_nthreads
                    )
                    # printf("End: calling the first barrier for the 4th loop; work_id_4th_loop: %d, comm_id_4th_loop: %d\n", work_id_4th_loop, comm_id_4th_loop)

                    gemm_pack_b(loop4_partition_b, loop4_partition_b_width,
                                loop4_partition_b_height, packed_b_buf,
                                comm_id_packb, work_id_packb, loop3_nthreads)

                    # The barrier at the end of the packing of B
                    # printf(
                    #     "Begin: calling the second barrier for the 4th loop; work_id_4th_loop: %d, comm_id_4th_loop: %d, work_id_5th_loop: %d, i_loop4: %d, loop5_partition_b_start_col: %d\n",
                    #     work_id_4th_loop, comm_id_4th_loop, work_id_5th_loop, i_loop4, loop5_partition_b_start_col)
                    thrcomm_barrier(
                        comm_id_packb,
                        ~packb_thrcomm_barrier_sense[work_id_5th_loop],
                        ~packb_thrcomm_barrier_threads_arrived[work_id_5th_loop],
                        packb_nthreads
                    )
                    # printf(
                    #     "End: calling the second barrier for the 4th loop; work_id_4th_loop: %d, comm_id_4th_loop: %d, work_id_5th_loop: %d, i_loop4: %d, loop5_partition_b_start_col: %d\n",
                    #     work_id_4th_loop, comm_id_4th_loop, work_id_5th_loop, i_loop4, loop5_partition_b_start_col)

                    # TODO: The loop3 and beyond should start here?
                    gemm_3rd_loop(
                        a, packed_b_buf, c,
                        loop4_partition_a_start_col,
                        loop4_partition_b_start_col,
                        loop4_partition_b_height,
                        loop4_partition_b_width,
                        comm_id_3rd_loop,
                        work_id_3rd_loop,
                        is_first,
                        work_id_4th_loop,
                        work_id_5th_loop
                    )

                    # # TODO: Is not adding this barrier at the end the problem?
                    # printf(
                    #     "Begin: calling the third barrier for the 4th loop; work_id_4th_loop: %d, comm_id_4th_loop: %d, work_id_5th_loop: %d, i_loop4: %d, loop5_partition_b_start_col: %d, work_id_5th_loop: %d\n",
                    #     work_id_4th_loop, comm_id_4th_loop, work_id_5th_loop, i_loop4, loop5_partition_b_start_col)
                    thrcomm_barrier(
                        comm_id_packb,
                        ~packb_thrcomm_barrier_sense[work_id_5th_loop],
                        ~packb_thrcomm_barrier_threads_arrived[work_id_5th_loop],
                        packb_nthreads
                    )
                    # printf(
                    #     "End: calling the third barrier for the 4th loop; work_id_4th_loop: %d, comm_id_4th_loop: %d, work_id_5th_loop: %d, i_loop4: %d, loop5_partition_b_start_col: %d\n",
                    #     work_id_4th_loop, comm_id_4th_loop, work_id_5th_loop, i_loop4, loop5_partition_b_start_col)

                    i_loop4 += b_alg_loop4

            gemm_4th_loop.kind = "cpu_internal"

            @hidet.script
            def gemm_5th_loop(a: float32[m_size, k_size],
                              b: float32[k_size, n_size],
                              c: float32[m_size, n_size],
                              work_id_5th_loop: int32,
                              comm_id_5th_loop: int32):
                comm_id_4th_loop = comm_id_5th_loop % loop4_nthreads
                work_id_4th_loop = comm_id_4th_loop // (loop4_nthreads // loop4_nways)

                loop5_my_start = -1
                loop5_my_end = -1
                thread_range_sub(loop5_nways, work_id_5th_loop, n_size,
                                 NR, ~loop5_my_start, ~loop5_my_end)

                loop5_iter = loop5_my_start
                while loop5_iter < loop5_my_end:
                    b_alg_loop5 = determine_blocksize_f_sub(loop5_iter,
                                                            loop5_my_end, NC)
                    b_alg_loop5 = min(b_alg_loop5, loop5_my_end - loop5_iter)

                    loop5_partition_c_width = b_alg_loop5
                    loop5_partition_c_start_col = loop5_iter
                    loop5_partition_b_width = b_alg_loop5,
                    loop5_partition_b_start_col = loop5_iter
                    #
                    # printf(
                    #     "work_id_5th_loop: %d, comm_id_5th_loop: %d, b_alg_loop5: %d, loop5_partition_b_width: %d, loop5_partition_b_start_col: %d, loop5_iter: %d, loop5_my_start: %d, loop5_my_end: %d\n\n",
                    #     work_id_5th_loop, comm_id_5th_loop, b_alg_loop5, loop5_partition_b_width,
                    #     loop5_partition_b_start_col, loop5_iter, loop5_my_start, loop5_my_end)
                    gemm_4th_loop(a, b, c,
                                  loop5_partition_b_width,
                                  loop5_partition_b_start_col,
                                  comm_id_4th_loop,
                                  work_id_4th_loop,
                                  work_id_5th_loop)
                    loop5_iter += b_alg_loop5

            gemm_5th_loop.kind = 'cpu_internal'

            ################### Start of the main kernel ###################
            @hidet.script
            def matmul_kernel_x86_v3(a: float32[m_size, k_size], b: float32[k_size, n_size],
                                     c: float32[m_size, n_size]):

                init_thr(packa_thrcomm_barrier_sense,
                         packa_thrcomm_threads_arrived,
                         loop3_nways)
                init_thr(packb_thrcomm_barrier_sense,
                         packb_thrcomm_barrier_threads_arrived,
                         loop5_nways)

                # the nthreads and nways print for each loop
                # printf("nthreads: %d\n", nthreads)
                # printf("loop5_nthreads: %d, loop5_nways: %d\n", loop5_nthreads, loop5_nways)
                # printf("loop4_nthreads: %d, loop4_nways: %d\n", loop4_nthreads, loop4_nways)
                # printf("loop3_nthreads: %d, loop3_nways: %d\n", loop3_nthreads, loop3_nways)
                # printf("macro_nthreads: %d, macro_nways: %d\n", macro_nthreads, macro_nways)
                # printf("loop1_nthreads: %d, loop1_nways: %d\n", loop1_nthreads, loop1_nways)
                #
                # printf("packb_nthreads: %d, packa_nthreads: %d\n", packb_nthreads, packa_nthreads)
                #
                # printf("packed_b_width: %d, packed_b_total_width: %d, packed_b_height: %d\n", packed_b_width,
                #        packed_b_total_width, packed_b_height)
                # printf("packed_a_width: %d, packed_a_individual_height: %d, packed_a_total_height: %d\n",
                #        packed_a_width, packed_a_individual_height, packed_a_total_height)
                #
                # printf("packed_b_total_size: %d, packed_a_total_size: %d\n", packed_b_total_size, packed_a_total_size)
                # printf("packed_b_individual_size: %d, packed_a_individual_size: %d\n", packed_b_individual_size, packed_a_individual_size)

                # for i in grid(m_size):
                #     for j in grid(n_size):
                #         c[i, j] = 0.0

                parallel_attr = 'p' + str(nthreads)
                # The outermost loop spawning threads
                for tidx in grid(nthreads, attrs=parallel_attr):
                    tid_5th_loop = tidx
                    work_id_5th_loop = tid_5th_loop // (nthreads // loop5_nways)
                    comm_id_5th_loop = tid_5th_loop
                    # printf("tidx: %d, tid_5th_loop: %d, work_id_5th_loop: %d, comm_id_5th_loop: %d\n", tidx, tid_5th_loop, work_id_5th_loop, comm_id_5th_loop)

                    gemm_5th_loop(a, b, c, work_id_5th_loop, comm_id_5th_loop)

            assert isinstance(matmul_kernel_x86_v3, hidet.ir.Function)
            matmul_kernel_x86_v3.kind = "cpu_kernel"
            ir_module = module.ir_module()
            return ir_module


class Matmulx86Op_refactored(Operator):
    def __init__(self, a: Tensor, b: Tensor):
        if not (len(a.shape) == len(b.shape) == 2 and a.shape[1] == b.shape[0]):
            raise ValueError('Matrix multiplication: incompatible sizes: {} and {}'.format(a.shape, b.shape))
        task = MatmulF32Taskx86_refactored(input_like(a, 'a'), input_like(b, 'b'))
        super().__init__(inputs=[a, b], attributes={}, task=task)


def matmul_x86_refactored(a: Tensor, b: Tensor) -> Tensor:
    return Matmulx86Op_refactored(a, b).outputs[0]
