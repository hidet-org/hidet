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
from typing import List
import hidet
from hidet.ir import IRModule
from hidet.ir.compute import reduce
from hidet.ir.layout import DataLayout
from hidet.ir.type import tensor_type
from hidet.ir import primitives as prim
from hidet.ir.primitives import active_mask, shfl_down_sync
from hidet.graph.ops.definitions.utils import tune
from hidet.lang import f16, f32, i32, u32, spatial, repeat, tensor
from hidet.lang import attr, grid, tensor_pointer, view, col_spatial
from hidet.lang.cuda import blockIdx, threadIdx, syncthreads, dynamic_shared_memory, register_tensor
from hidet.graph.ops.definitions.utils import Task, Operator, Tensor, TensorNode, compute, input_like
from hidet.graph.ops.definitions.utils import broadcast_shape, broadcast_shapes, broadcast_indices
from hidet.utils.py import cdiv, prod


class AttnMaskAddTask(Task):
    def __init__(self, name: str, q: TensorNode, k: TensorNode, v: TensorNode, mask: TensorNode):
        q_shape = q.const_shape()
        k_shape = k.const_shape()
        v_shape = v.const_shape()
        mask_shape = mask.const_shape()
        n_size = q_shape[-2]
        d_size = q_shape[-1]
        o_shape = broadcast_shapes([q_shape[:-2], k_shape[:-2], v_shape[:-2]]) + [n_size, d_size]
        o_head, q_head, k_head, v_head = (o_shape[:-2], q_shape[:-2], k_shape[:-2], v_shape[:-2])
        qk_head = broadcast_shape(q_head, k_head)
        mask_shape = mask.const_shape()

        qk = compute(
            name='qk',
            shape=qk_head + [n_size, n_size],
            fcompute=lambda *indices: reduce(
                shape=[d_size],
                fcompute=lambda d: q[broadcast_indices(indices[:-2], q_head, qk_head) + [indices[-2], d]]
                * k[broadcast_indices(indices[:-2], k_head, qk_head) + [d, indices[-1]]],
                reduce_type='sum',
            ),
        )

        qk_shape = qk.const_shape()

        qk_masked = compute(
            name='qk_masked',
            shape=qk_shape,
            fcompute=lambda *indices: mask[broadcast_indices(indices, mask_shape, qk_shape)] + qk[indices],
        )

        axis = len(qk_shape) - 1
        axis_extent = qk_shape[axis]
        reduced_shape = qk_shape[:axis] + qk_shape[axis + 1 :]

        # max value
        max_value = compute(
            name='max_value',
            shape=reduced_shape,
            fcompute=lambda *indices: reduce(
                shape=[axis_extent],
                fcompute=lambda a: qk_masked[indices[:axis] + (a,) + indices[axis:]],
                reduce_type='max',
            ),
        )

        # exp
        exp_value = compute(
            name='exp_value',
            shape=qk_shape,
            fcompute=lambda *indices: prim.exp(qk_masked[indices] - max_value[indices[:axis] + indices[axis + 1 :]]),
        )

        # sum
        sum_value = compute(
            name='sum_value',
            shape=reduced_shape,
            fcompute=lambda *indices: reduce(
                shape=[axis_extent],
                fcompute=lambda a: exp_value[indices[:axis] + (a,) + indices[axis:]],
                reduce_type='sum',
            ),
        )

        sm = compute(
            name='sm',
            shape=qk_shape,
            fcompute=lambda *indices: exp_value[indices] / sum_value[indices[:axis] + indices[axis + 1 :]],
        )
        o = compute(
            name='o',
            shape=o_shape,
            fcompute=lambda *indices: reduce(
                shape=[n_size],
                fcompute=lambda a: sm[broadcast_indices(indices[:-2], qk_head, o_head) + [indices[-2], a]]
                * v[broadcast_indices(indices[:-2], v_head, o_head) + [a, indices[-1]]],
                reduce_type='sum',
            ),
        )

        super().__init__(name=name, inputs=[q, k, v, mask], outputs=[o])

    def allow_prologue(self) -> bool:
        return False

    def allow_epilogue(self) -> bool:
        return False

    def implement_cuda(self, working_dir: str) -> IRModule:
        return tune.extract_ir_modules(self.cuda_schedule_attn)

    @tune.space(2, 'block_size', [128, 256])
    @tune.space(2, 'block_j', [128, 256])
    @tune.space(2, 'block_i', [16, 32])
    @tune.space(2, 'i_split', [s for s in range(1, 31)])
    @tune.space(0, 'block_size', [128])
    @tune.space(0, 'block_j', [256])
    @tune.space(0, 'block_i', [16])
    @tune.space(0, 'i_split', [2])
    def cuda_schedule_attn(self, i_split=2, block_i=16, block_j=256, block_size=128) -> IRModule:
        def calc_swizzle_size(d):
            powers_of_two = [128, 64, 32, 16, 8]
            for n in powers_of_two:
                if d == n:
                    return d, 1
                if d % n == 0:
                    return n, d // n
            return -1, -1

        task = self
        node_q, node_k, node_v, node_mask, node_o = (
            task.inputs[0],
            task.inputs[1],
            task.inputs[2],
            task.inputs[3],
            task.outputs[0],
        )
        q_shape: List[int] = node_q.const_shape()
        k_shape: List[int] = node_k.const_shape()
        v_shape: List[int] = node_v.const_shape()
        o_shape: List[int] = node_o.const_shape()
        mask_shape: List[int] = node_mask.const_shape()
        o_head, q_head, k_head, v_head = (o_shape[:-2], q_shape[:-2], k_shape[:-2], v_shape[:-2])
        qk_head = broadcast_shape(q_head, k_head)
        bs_qk = prod(qk_head)
        bs = prod(o_head)
        assert bs == bs_qk

        from hidet.lang.cuda import MmaConfig, mma_sync, cp_async, ldmatrix, cp_async_wait_all

        local_layout = DataLayout.local
        row_major = DataLayout.row_major
        n_size = q_shape[-2]
        d_size = q_shape[-1]
        dtype = task.inputs[0].ttype.dtype
        dtype_size = dtype.nbytes
        warp_size = 32
        tune.check(d_size % 8 == 0)
        tune.check(d_size <= 160)
        tune.check(n_size >= 64)
        block_j = min(block_j, n_size)

        acc_dtype = f16
        sm_dtype = f32
        mma_configs = {'m16n8k16f32': MmaConfig.m16n8k16_f16_f32(), 'm16n8k16f16': MmaConfig.m16n8k16_f16_f16()}
        mma_config = mma_configs['m16n8k16f16'] if acc_dtype == f16 else mma_configs['m16n8k16f32']
        # mma_config = MmaConfig.m16n8k8_f16_f16()
        mma_m = mma_config.m
        mma_n = mma_config.n
        mma_k = mma_config.k

        block_k = 32 * cdiv(d_size, 32)
        swizzle_unit, swizzle_repeat = calc_swizzle_size(block_k)
        tune.check(swizzle_repeat > 0)

        num_warps = block_size // warp_size

        # Number of elements each warp handles in a dimension. 16, 64, 64
        warp_elems_m, warp_elems_n, warp_elems_k = block_i, block_j // num_warps, mma_k * cdiv(block_k, mma_k)
        # Number of warps in each dimension. 1, 4, 1
        warp_count_m, warp_count_n, warp_count_k = (
            block_i // warp_elems_m,
            block_j // warp_elems_n,
            cdiv(block_k, warp_elems_k),
        )
        # Number of m16n8k16 mma's each warp performs in each dim. 1, 8, 4
        mmas_per_warp_m, mmas_per_warp_n, mmas_per_warp_k = (
            warp_elems_m // mma_m,
            warp_elems_n // mma_n,
            warp_elems_k // mma_k,
        )

        # Number of warps in each dimension. 1, 4, 1
        warp_count_m_o, warp_count_k_o = warp_count_m, 1
        warp_count_n_o = num_warps // (warp_count_m_o * warp_count_k_o)
        # Number of elements each warp handles in a dimension. 16, 16, 256
        assert block_k % warp_count_n_o == 0
        warp_elems_m_o, warp_elems_n_o, warp_elems_k_o = (
            block_i // warp_count_m_o,
            mma_n * cdiv(block_k // warp_count_n_o, mma_n),
            block_j // warp_count_k_o,
        )
        # Number of m16n8k16 mma's each warp performs in each dim. 1, 2, 16
        mmas_per_warp_m_o, mmas_per_warp_n_o, mmas_per_warp_k_o = (
            warp_elems_m_o // mma_m,
            warp_elems_n_o // mma_n,
            warp_elems_k_o // mma_k,
        )

        tune.check(dtype_size == 2)
        tune.check(block_j % warp_size == 0)
        tune.check(block_j % num_warps == 0)
        tune.check((block_i * block_k) % block_size == 0)
        tune.check((block_k * block_j) % block_size == 0)
        tune.check(block_j % (64) == 0)
        tune.check(block_size >= block_k)

        n_tiles = (n_size + block_i - 1) // block_i
        i_tiles_per_tb = cdiv(n_tiles, i_split)
        i_rows_per_tb = i_tiles_per_tb * block_i
        j_tiles = (n_size + block_j - 1) // block_j

        smem_bytes_q = dtype_size * block_i * block_k
        smem_bytes_k = dtype_size * block_k * block_j
        smem_bytes_v = 0
        smem_bytes_qk = 0
        smem_bytes_l = sm_dtype.nbytes * i_rows_per_tb
        smem_bytes_m = sm_dtype.nbytes * i_rows_per_tb
        smem_bytes_lij = sm_dtype.nbytes * block_i
        smem_bytes_mij = sm_dtype.nbytes * block_i
        tune.check(dtype_size * block_i * block_j <= smem_bytes_k)  # smem_bytes_qk <= smem_bytes_k

        smem_bytes_offsets = {
            'q': 0,
            'o': 0,
            'k': smem_bytes_q,
            # 'v': smem_bytes_q + smem_bytes_k,
            'v': smem_bytes_q,
            # 'qk': smem_bytes_q + smem_bytes_k + smem_bytes_v,
            'qk': smem_bytes_q,
            'l': smem_bytes_q + smem_bytes_k + smem_bytes_v + smem_bytes_qk,
            'm': smem_bytes_q + smem_bytes_k + smem_bytes_v + smem_bytes_qk + smem_bytes_l,
            'lij': smem_bytes_q + smem_bytes_k + smem_bytes_v + smem_bytes_qk + smem_bytes_l + smem_bytes_m,
            'mij': smem_bytes_q
            + smem_bytes_k
            + smem_bytes_v
            + smem_bytes_qk
            + smem_bytes_l
            + smem_bytes_m
            + smem_bytes_lij,
        }

        dynamic_smem_bytes = (
            smem_bytes_q
            + smem_bytes_k
            + smem_bytes_qk
            + smem_bytes_v
            + smem_bytes_l
            + smem_bytes_m
            + smem_bytes_lij
            + smem_bytes_mij
        )
        used_smem_bytes_per_block = dynamic_smem_bytes
        tune.check(used_smem_bytes_per_block <= 99000)

        smem_l_type = tensor_type(sm_dtype, shape=[i_rows_per_tb])
        smem_m_type = tensor_type(sm_dtype, shape=[i_rows_per_tb])
        smem_lij_type = tensor_type(sm_dtype, shape=[block_i])
        smem_mij_type = tensor_type(sm_dtype, shape=[block_i])

        smem_o_layout = (
            row_major((1, swizzle_repeat)) * row_major((block_i, swizzle_unit // 8)).swizzle(1) * row_major((1, 8))
        )
        # smem_o_layout = row_major((block_i, block_k))

        smem_k_layout = row_major((block_k // 8, block_j // 64)) * row_major((8, 8)).swizzle(1) * row_major((1, 8))
        # smem_k_layout = row_major((block_k, block_j))
        smem_qk_layout = row_major((block_i, block_j // 8)).swizzle(1) * row_major((1, 8))
        # smem_qk_layout = row_major((block_i, block_j))
        # smem_v_layout = (
        #     row_major((block_j // 8, block_k // 64))
        #     * row_major((8,8)).swizzle(1)
        #     * row_major((1,8))
        # )
        smem_v_layout = row_major((block_j // 8, block_k // 8)) * row_major((8, 1)).swizzle(1) * row_major((1, 8))
        # smem_v_layout = row_major((block_j, block_k))

        smem_o_type = tensor_type('float16', shape=[block_i, block_k], layout=smem_o_layout)
        smem_q_type = smem_o_type
        smem_k_type = tensor_type('float16', shape=[block_k, block_j], layout=smem_k_layout)
        smem_qk_type = tensor_type('float16', shape=[block_i, block_j], layout=smem_qk_layout)
        smem_v_type = tensor_type('float16', shape=[block_j, block_k], layout=smem_v_layout)

        n_size_per_thread = (i_rows_per_tb + block_size - 1) // block_size
        lm_layout = repeat(n_size_per_thread) * spatial(min(i_rows_per_tb, block_size))

        rows_per_thread_mma_o = 2 * mmas_per_warp_m_o
        # regs_li_new should be 2x1, it is accessed via [r] ([~16])
        regs_li_new_layout = row_major((rows_per_thread_mma_o, 1)) * local_layout((block_i // rows_per_thread_mma_o, 1))
        regs_mi_new_layout = regs_li_new_layout
        regs_exp_mij_layout = regs_li_new_layout

        # Round up to nearest multiple of 8
        q_elems_per_thread = block_i * block_k // block_size
        q_elems_per_thread = 8 * cdiv(q_elems_per_thread, 8)

        o_s2g_layout = spatial(cdiv(block_i, q_elems_per_thread), block_k) * repeat(
            q_elems_per_thread, 1
        )  # 16 x 64, 8 elems per thread
        t_per_block_k_8_floor = block_size // (block_k // 8)
        if block_i < t_per_block_k_8_floor:
            q_g2s_layout = spatial(block_i, block_k // 8)
        else:
            q_g2s_layout = repeat(cdiv(block_i, t_per_block_k_8_floor), 1) * spatial(
                t_per_block_k_8_floor, block_k // 8
            )
        k_g2s_layout = repeat(cdiv(block_k, (block_size // (block_j // 8))), 1) * spatial(
            block_size // (block_j // 8), block_j // 8
        )
        v_g2s_layout = repeat(cdiv(block_j, t_per_block_k_8_floor), 1) * spatial(t_per_block_k_8_floor, block_k // 8)
        o_g2s_layout = q_g2s_layout

        with hidet.script_module() as module:

            @hidet.script
            def resolve_ldmatrix(regs: ~f16, smem_addr: ~f16, is_A: hidet.lang.boolean):
                if mma_k == 16:
                    if is_A:
                        b32_regs = view(regs, u32[4])
                        ldmatrix(
                            regs=[b32_regs[0], b32_regs[1], b32_regs[2], b32_regs[3]],
                            smem_addr=smem_addr,
                            shared_space_addr=False,
                            trans=False,
                        )
                    else:
                        b32_regs = view(regs, u32[2])
                        ldmatrix(regs=[b32_regs[0], b32_regs[1]], smem_addr=smem_addr, trans=True)
                elif mma_k == 8:
                    if is_A:
                        b32_regs = view(regs, u32[2])
                        ldmatrix(
                            regs=[b32_regs[0], b32_regs[1]], smem_addr=smem_addr, shared_space_addr=False, trans=False
                        )
                    else:
                        b32_regs = view(regs, u32[1])
                        ldmatrix(regs=[b32_regs[0]], smem_addr=smem_addr, trans=True)

            @hidet.script
            def init_o_gmem(o: f16[o_head + [n_size, d_size]], offset_i: i32):
                o_head_index = spatial(*o_head).map(blockIdx.y)
                gmem_o = o[o_head_index][offset_i:, :]
                for i, j in o_s2g_layout.on(threadIdx.x):
                    if threadIdx.x < o_s2g_layout.num_workers and i < smem_o_type.shape[0]:
                        gmem_o.write([i, j], f16.zero, protected=True)

            @hidet.script
            def init_lm_smem(smem_l: smem_l_type, smem_m: smem_m_type):
                for i in lm_layout.on(threadIdx.x):
                    if i < smem_l_type.shape[0]:
                        smem_l[i] = smem_l_type.dtype.zero
                        smem_m[i] = smem_m_type.dtype.min_value

            @hidet.script
            def copy_k_g2s(k: f16[k_head + [d_size, n_size]], smem_k: smem_k_type, offset_j: i32):
                o_head_index = spatial(*o_head).map(blockIdx.y)
                gmem_k = k[broadcast_indices(o_head_index, k_head, o_head)][:, offset_j:]
                for i, j_seg in k_g2s_layout.on(threadIdx.x):
                    j = j_seg * 8
                    src_size = 0 if (i >= d_size or offset_j + j >= n_size) else min(n_size - j, 8)
                    if threadIdx.x < k_g2s_layout.num_workers and i < smem_k_type.shape[0]:
                        cp_async(~smem_k[i, j], ~gmem_k[i, j], cp_size=16, src_size=src_size * 2, cache_level='global')

            @hidet.script
            def copy_v_g2s(v: f16[v_head + [n_size, d_size]], smem_v: smem_v_type, offset_j: i32):
                o_head_index = spatial(*o_head).map(blockIdx.y)
                gmem_v = v[broadcast_indices(o_head_index, v_head, o_head)][offset_j:, :]
                for i, j_seg in v_g2s_layout.on(threadIdx.x):
                    j = j_seg * 8
                    src_size = 0 if (offset_j + i >= n_size or j >= d_size) else min(d_size - j, 8)
                    if threadIdx.x < v_g2s_layout.num_workers and i < smem_v_type.shape[0]:
                        cp_async(~smem_v[i, j], ~gmem_v[i, j], cp_size=16, src_size=src_size * 2, cache_level='global')

            @hidet.script
            def copy_q_g2s(q: f16[q_head + [n_size, d_size]], smem_q: smem_q_type, offset_i: i32):
                o_head_index = spatial(*o_head).map(blockIdx.y)
                gmem_q = q[broadcast_indices(o_head_index, q_head, o_head)][offset_i:, :]
                for i, j_seg in q_g2s_layout.on(threadIdx.x):
                    j = j_seg * 8
                    src_size = 0 if (offset_i + i >= n_size or j >= d_size) else min(d_size - j, 8)
                    if threadIdx.x < q_g2s_layout.num_workers and i < smem_q_type.shape[0]:
                        cp_async(~smem_q[i, j], ~gmem_q[i, j], cp_size=16, src_size=src_size * 2, cache_level='global')

            @hidet.script
            def copy_o_g2s(o: f16[o_head + [n_size, d_size]], smem_o: smem_o_type, offset_i: i32):
                o_head_index = spatial(*o_head).map(blockIdx.y)
                gmem_o = o[o_head_index][offset_i:, :]
                for i, j_seg in o_g2s_layout.on(threadIdx.x):
                    j = j_seg * 8
                    src_size = 0 if (offset_i + i >= n_size or j >= d_size) else min(d_size - j, 8)
                    if threadIdx.x < o_g2s_layout.num_workers and i < smem_o_type.shape[0]:
                        cp_async(~smem_o[i, j], ~gmem_o[i, j], cp_size=16, src_size=src_size * 2, cache_level='global')

            @hidet.script
            def copy_o_s2g(o: f16[o_head + [n_size, d_size]], smem_o: smem_o_type, offset_i: i32):
                o_head_index = spatial(*o_head).map(blockIdx.y)
                gmem_o = o[o_head_index][offset_i:, :]
                for i, j in o_s2g_layout.on(threadIdx.x):
                    if threadIdx.x < o_s2g_layout.num_workers and i < smem_o_type.shape[0]:
                        gmem_o.write([i, j], smem_o[i, j], protected=True)

            @hidet.script
            def copy_q_s2r(mma_i: int, k1: int, regs_q: f16[mma_config.a_elements], smem_q: smem_q_type):
                warp_id, lane_id = threadIdx.x / 32, threadIdx.x % 32
                for wi, _, wk in spatial(warp_count_m, warp_count_n, warp_count_k).on(warp_id):
                    p, q = col_spatial(16, 2).map(lane_id)
                    row_addr = ~smem_q[wi * warp_elems_m + mma_i * mma_m + p, wk * warp_elems_k + k1 * mma_k + q * 8]
                    resolve_ldmatrix(regs_q, row_addr, True)

            @hidet.script
            def copy_k_s2r(mma_j: int, k1: int, regs_k: f16[mma_config.b_elements], smem_k: smem_k_type):
                warp_id, lane_id = threadIdx.x / 32, threadIdx.x % 32
                for _, wj, wk in spatial(warp_count_m, warp_count_n, warp_count_k).on(warp_id):
                    p = col_spatial(16, 2).map(lane_id)[0]
                    row_addr = ~smem_k[wk * warp_elems_k + k1 * mma_k + p, wj * warp_elems_n + mma_j * mma_n]
                    resolve_ldmatrix(regs_k, row_addr, False)

            @hidet.script
            def copy_qk_s2r(mma_i: int, k1: int, regs_qk: f16[mma_config.a_elements], smem_qk: smem_qk_type):
                warp_id, lane_id = threadIdx.x / 32, threadIdx.x % 32
                for wi, _, wk in spatial(warp_count_m_o, warp_count_n_o, warp_count_k_o).on(warp_id):
                    if not warp_id >= spatial(warp_count_m_o, warp_count_n_o, warp_count_k_o).num_workers:
                        p, q = col_spatial(16, 2).map(lane_id)
                        row_addr = ~smem_qk[
                            wi * warp_elems_m_o + mma_i * mma_m + p, wk * warp_elems_k_o + k1 * mma_k + q * 8
                        ]
                        resolve_ldmatrix(regs_qk, row_addr, True)

            @hidet.script
            def copy_v_s2r(mma_j: int, k1: int, regs_v: f16[mma_config.b_elements], smem_v: smem_v_type):
                warp_id, lane_id = threadIdx.x / 32, threadIdx.x % 32
                for _, wj, wk in spatial(warp_count_m_o, warp_count_n_o, warp_count_k_o).on(warp_id):
                    if not warp_id >= spatial(warp_count_m_o, warp_count_n_o, warp_count_k_o).num_workers:
                        p = col_spatial(16, 2).map(lane_id)[0]
                        row_addr = ~smem_v[wk * warp_elems_k_o + k1 * mma_k + p, wj * warp_elems_n_o + mma_j * mma_n]
                        resolve_ldmatrix(regs_v, row_addr, False)

            @hidet.script
            def qk_softmax_reduce(
                smem_qk: smem_qk_type,
                smem_mij: smem_mij_type,
                smem_lij: smem_lij_type,
                regs_acc: acc_dtype[mmas_per_warp_m, mmas_per_warp_n, mma_config.c_elements],
            ):
                # Everything below assumes m16n8k16_f16 layout
                warp_mask = active_mask()
                warp_id, lane_id = threadIdx.x / 32, threadIdx.x % 32
                # Each thread holds c elements in 2 rows in mma
                rv = register_tensor(acc_dtype, [2])

                # Reduce mij
                rv[0] = acc_dtype.min_value
                rv[1] = acc_dtype.min_value
                # In most cases, wi = wk = 0, and wj is 0 to warp_count_m(=num_warps)
                wi, wj, _ = spatial(warp_count_m, warp_count_n, warp_count_k).on(warp_id)[0]
                c_map = repeat(2, 1) * spatial(8, 4)
                for mma_i in range(mmas_per_warp_m):
                    rv[0] = acc_dtype.min_value
                    rv[1] = acc_dtype.min_value
                    for mma_j in range(mmas_per_warp_n):
                        rv[0] = prim.max(rv[0], regs_acc[mma_i, mma_j, 0])
                        rv[0] = prim.max(rv[0], regs_acc[mma_i, mma_j, 1])
                        rv[1] = prim.max(rv[1], regs_acc[mma_i, mma_j, 2])
                        rv[1] = prim.max(rv[1], regs_acc[mma_i, mma_j, 3])
                    rv[0] = prim.max(rv[0], shfl_down_sync(warp_mask, rv[0], 2, 4))
                    rv[0] = prim.max(rv[0], shfl_down_sync(warp_mask, rv[0], 1, 2))
                    rv[1] = prim.max(rv[1], shfl_down_sync(warp_mask, rv[1], 2, 4))
                    rv[1] = prim.max(rv[1], shfl_down_sync(warp_mask, rv[1], 1, 2))
                    for n_round in range(warp_count_n):
                        if n_round == wj:
                            if threadIdx.x % 4 == 0:
                                for i, j in c_map.on(lane_id):
                                    delta_m = wi * warp_elems_m + mma_i * mma_m + i
                                    if n_round == 0:
                                        smem_mij[delta_m] = rv[i // 8]
                                    else:
                                        smem_mij[delta_m] = prim.max(smem_mij[delta_m], rv[i // 8])
                        syncthreads()

                # Softmax
                for mma_i, mma_j in grid(mmas_per_warp_m, mmas_per_warp_n):
                    p = 0
                    for i, j in mma_config.c_store_map.on(lane_id):
                        delta_m = wi * warp_elems_m + mma_i * mma_m + i
                        delta_n = wj * warp_elems_n + mma_j * mma_n + j
                        regs_acc[mma_i, mma_j, p] = prim.exp(regs_acc[mma_i, mma_j, p] - smem_mij[delta_m])
                        smem_qk[delta_m, delta_n] = regs_acc[mma_i, mma_j, p]
                        p += 1
                syncthreads()

                # Reduce lij
                rv[0] = acc_dtype.zero
                rv[1] = acc_dtype.zero
                for mma_i in range(mmas_per_warp_m):
                    rv[0] = acc_dtype.zero
                    rv[1] = acc_dtype.zero
                    for mma_j in range(mmas_per_warp_n):
                        rv[0] = rv[0] + regs_acc[mma_i, mma_j, 0]
                        rv[0] = rv[0] + regs_acc[mma_i, mma_j, 1]
                        rv[1] = rv[1] + regs_acc[mma_i, mma_j, 2]
                        rv[1] = rv[1] + regs_acc[mma_i, mma_j, 3]
                    rv[0] = rv[0] + shfl_down_sync(warp_mask, rv[0], 2, 4)
                    rv[0] = rv[0] + shfl_down_sync(warp_mask, rv[0], 1, 2)
                    rv[1] = rv[1] + shfl_down_sync(warp_mask, rv[1], 2, 4)
                    rv[1] = rv[1] + shfl_down_sync(warp_mask, rv[1], 1, 2)
                    for n_round in range(warp_count_n):
                        if n_round == wj:
                            if threadIdx.x % 4 == 0:
                                for i, j in c_map.on(lane_id):
                                    delta_m = wi * warp_elems_m + mma_i * mma_m + i
                                    if n_round == 0:
                                        smem_lij[delta_m] = rv[i // 8]
                                    else:
                                        smem_lij[delta_m] = smem_lij[delta_m] + rv[i // 8]
                        syncthreads()

            @hidet.script
            def warp_mma(
                regs_a: f16[mma_config.a_elements],
                regs_b: f16[mma_config.b_elements],
                regs_c: acc_dtype[mma_config.c_elements],
            ):
                mma_sync(mma_config, regs_a, regs_b, regs_c)

            @hidet.script
            def attn_kernel(
                q: f16[q_head + [n_size, d_size]],
                k: f16[k_head + [d_size, n_size]],
                v: f16[v_head + [n_size, d_size]],
                mask: f16[mask_shape],
                o: f16[o_head + [n_size, d_size]],
            ):
                attr.cuda_grid_dim = (i_split, bs)
                attr.cuda_block_dim = block_size
                attr.cuda_min_blocks = 1
                attr.cuda_dynamic_smem_bytes = dynamic_smem_bytes

                offset_n = blockIdx.x * i_rows_per_tb

                smem_q = tensor_pointer('float16', shape=smem_q_type.shape, layout=smem_q_type.layout)
                smem_o = tensor_pointer('float16', shape=smem_o_type.shape, layout=smem_o_type.layout)
                smem_k = tensor_pointer('float16', shape=smem_k_type.shape, layout=smem_k_type.layout)
                smem_qk = tensor_pointer('float16', shape=smem_qk_type.shape, layout=smem_qk_type.layout)
                smem_v = tensor_pointer('float16', shape=smem_v_type.shape, layout=smem_v_type.layout)
                smem_l = tensor_pointer(smem_l_type.dtype, shape=smem_l_type.shape)
                smem_m = tensor_pointer(smem_m_type.dtype, shape=smem_m_type.shape)
                smem_lij = tensor_pointer(smem_lij_type.dtype, shape=smem_lij_type.shape)
                smem_mij = tensor_pointer(smem_mij_type.dtype, shape=smem_mij_type.shape)

                smem_q = dynamic_shared_memory(byte_offset=smem_bytes_offsets['q'], dtype=f16)
                smem_o = dynamic_shared_memory(byte_offset=smem_bytes_offsets['o'], dtype=f16)
                smem_k = dynamic_shared_memory(byte_offset=smem_bytes_offsets['k'], dtype=f16)
                smem_qk = dynamic_shared_memory(byte_offset=smem_bytes_offsets['qk'], dtype=f16)
                smem_v = dynamic_shared_memory(byte_offset=smem_bytes_offsets['v'], dtype=f16)
                smem_l = dynamic_shared_memory(byte_offset=smem_bytes_offsets['l'], dtype=smem_l_type.dtype)
                smem_m = dynamic_shared_memory(byte_offset=smem_bytes_offsets['m'], dtype=smem_m_type.dtype)
                smem_lij = dynamic_shared_memory(byte_offset=smem_bytes_offsets['lij'], dtype=smem_lij_type.dtype)
                smem_mij = dynamic_shared_memory(byte_offset=smem_bytes_offsets['mij'], dtype=smem_mij_type.dtype)

                regs_q = tensor('register', dtype='float16', shape=[mmas_per_warp_m, mma_config.a_elements])
                regs_k = tensor(
                    'register', dtype='float16', shape=[mmas_per_warp_k, mmas_per_warp_n, mma_config.b_elements]
                )
                regs_acc = tensor(
                    'register', dtype=acc_dtype, shape=[mmas_per_warp_m, mmas_per_warp_n, mma_config.c_elements]
                )
                regs_qk = tensor('register', dtype='float16', shape=[mmas_per_warp_m_o, mma_config.a_elements])
                regs_v = tensor(
                    'register', dtype='float16', shape=[mmas_per_warp_k_o, mmas_per_warp_n_o, mma_config.b_elements]
                )
                regs_acc_o = tensor(
                    'register', dtype=acc_dtype, shape=[mmas_per_warp_m_o, mmas_per_warp_n_o, mma_config.c_elements]
                )
                regs_li_new = tensor('register', dtype=smem_l_type.dtype, layout=regs_li_new_layout)
                regs_mi_new = tensor('register', dtype=smem_m_type.dtype, layout=regs_mi_new_layout)
                regs_exp_mij = tensor('register', dtype=smem_mij_type.dtype, layout=regs_exp_mij_layout)

                warp_id, lane_id = threadIdx.x / 32, threadIdx.x % 32
                for i in range(i_tiles_per_tb):
                    offset_i = offset_n + i * block_i
                    init_o_gmem(o, offset_i)
                init_lm_smem(smem_l, smem_m)

                for j in range(j_tiles):
                    # Load Kj, Vj into Smem
                    offset_j = block_j * j  # 256j
                    copy_v_g2s(v, smem_v, offset_j)
                    cp_async_wait_all()
                    syncthreads()
                    for mma_k in range(mmas_per_warp_k_o):
                        for mma_j in range(mmas_per_warp_n_o):
                            copy_v_s2r(mma_j, mma_k, ~regs_v[mma_k, mma_j, 0], smem_v)
                    syncthreads()
                    copy_k_g2s(k, smem_k, offset_j)
                    cp_async_wait_all()
                    syncthreads()
                    for mma_k in range(mmas_per_warp_k):
                        for mma_j in range(mmas_per_warp_n):
                            copy_k_s2r(mma_j, mma_k, ~regs_k[mma_k, mma_j, 0], smem_k)
                    for i in range(i_tiles_per_tb):
                        # Load Qi into Smem
                        offset_i = offset_n + i * block_i
                        copy_q_g2s(q, smem_q, offset_i)
                        # Compute QK = Qi * Kj
                        # Init regs_acc to 0
                        for a, b, c in grid(mmas_per_warp_m, mmas_per_warp_n, mma_config.c_elements):
                            regs_acc[a, b, c] = acc_dtype.zero
                        cp_async_wait_all()
                        syncthreads()
                        for mma_k in range(mmas_per_warp_k):
                            for mma_i in range(mmas_per_warp_m):
                                copy_q_s2r(mma_i, mma_k, ~regs_q[mma_i, 0], smem_q)
                            for mma_i, mma_j in grid(mmas_per_warp_m, mmas_per_warp_n):
                                warp_mma(~regs_q[mma_i, 0], ~regs_k[mma_k, mma_j, 0], ~regs_acc[mma_i, mma_j, 0])
                        # Apply Masking
                        qk_head_index = list(spatial(*qk_head).map(blockIdx.y))
                        for mma_i, mma_j in grid(mmas_per_warp_m, mmas_per_warp_n):
                            wi, wj, wk = spatial(warp_count_m, warp_count_n, warp_count_k).on(warp_id)[0]
                            p = 0
                            for ti, tj in mma_config.c_store_map.on(lane_id):
                                delta_m = offset_i + wi * warp_elems_m + mma_i * mma_m + ti
                                delta_n = offset_j + wj * warp_elems_n + mma_j * mma_n + tj
                                regs_acc[mma_i, mma_j, p] += mask[
                                    broadcast_indices(
                                        qk_head_index + [delta_m, delta_n], mask_shape, qk_head + [n_size, n_size]
                                    )
                                ]
                                p += 1
                        qk_softmax_reduce(smem_qk, smem_mij, smem_lij, regs_acc)
                        # Load Oi into Smem
                        copy_o_g2s(o, smem_o, offset_i)
                        for a, b, c in grid(mmas_per_warp_m_o, mmas_per_warp_n_o, mma_config.c_elements):
                            regs_acc_o[a, b, c] = acc_dtype.zero
                        for mma_k in range(mmas_per_warp_k_o):
                            for mma_i in range(mmas_per_warp_m_o):
                                copy_qk_s2r(mma_i, mma_k, ~regs_qk[mma_i, 0], smem_qk)
                            for mma_i, mma_j in grid(mmas_per_warp_m_o, mmas_per_warp_n_o):
                                warp_mma(~regs_qk[mma_i, 0], ~regs_v[mma_k, mma_j, 0], ~regs_acc_o[mma_i, mma_j, 0])
                        cp_async_wait_all()
                        syncthreads()
                        offset_lm_i = i * block_i
                        for k_round in range(warp_count_k):
                            for wi, wj, wk in spatial(warp_count_m_o, warp_count_n_o, warp_count_k_o).on(warp_id):
                                if wk == k_round:
                                    for mma_i, mma_j in grid(mmas_per_warp_m_o, 1):
                                        c_store_map = repeat(2, 1) * spatial(8, 4)
                                        for ti, _ in c_store_map.on(lane_id):
                                            delta_m = wi * warp_elems_m_o + mma_i * mma_m + ti
                                            mi = smem_m[offset_lm_i + delta_m]
                                            mij = smem_mij[delta_m]
                                            li = smem_l[offset_lm_i + delta_m]
                                            lij = smem_lij[delta_m]
                                            syncthreads()
                                            regs_mi_new[delta_m, 0] = prim.max(mi, mij)
                                            smem_m[offset_lm_i + delta_m] = regs_mi_new[delta_m, 0]
                                            exp_mi = prim.exp(mi - regs_mi_new[delta_m, 0])
                                            exp_mij = prim.exp(mij - regs_mi_new[delta_m, 0])
                                            # reuse regs_mi_new
                                            regs_mi_new[delta_m, 0] = exp_mi * li
                                            regs_li_new[delta_m, 0] = exp_mi * li + exp_mij * lij
                                            smem_l[offset_lm_i + delta_m] = regs_li_new[delta_m, 0]
                                            regs_exp_mij[delta_m, 0] = exp_mij
                                            syncthreads()

                        for k_round in range(warp_count_k):
                            for wi, wj, wk in spatial(warp_count_m_o, warp_count_n_o, warp_count_k_o).on(warp_id):
                                if wk == k_round:
                                    for mma_i, mma_j in grid(mmas_per_warp_m_o, mmas_per_warp_n_o):
                                        p = 0
                                        for ti, tj in mma_config.c_store_map.on(lane_id):
                                            delta_m = wi * warp_elems_m_o + mma_i * mma_m + ti
                                            delta_n = wj * warp_elems_n_o + mma_j * mma_n + tj
                                            smem_o[delta_m, delta_n] = (
                                                regs_mi_new[delta_m, 0] * smem_o[delta_m, delta_n]
                                                + regs_exp_mij[delta_m, 0] * regs_acc_o[mma_i, mma_j, p]
                                            ) / regs_li_new[delta_m, 0]
                                            # smem_o[delta_m, delta_n] +=  regs_acc_o[mma_i, mma_j, p]
                                            p += 1

                        syncthreads()
                        copy_o_s2g(o, smem_o, offset_i)
                        syncthreads()

        ir_module = module.ir_module()
        return ir_module


class AttnMaskAddOp(Operator):
    def __init__(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor):
        super().__init__(
            inputs=[q, k, v, mask],
            task=AttnMaskAddTask(
                'attn_mask_add', input_like(q, 'q'), input_like(k, 'k'), input_like(v, 'v'), input_like(mask, 'mask')
            ),
            attributes={},
        )
