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
from typing import List, Optional, Union
import hidet
from hidet.ir import IRModule
from hidet.ir.compute import reduce
from hidet.ir.layout import DataLayout
from hidet.ir.type import tensor_type
from hidet.ir import primitives as prim
from hidet.ir.primitives import active_mask, shfl_down_sync
from hidet.graph.ops.utils import tune
from hidet.lang import f16, f32, i32, u32, spatial, repeat, tensor
from hidet.lang import attrs, grid, tensor_pointer, view, col_spatial
from hidet.lang.cuda import blockIdx, threadIdx, syncthreads, dynamic_shared_memory, register_tensor
from hidet.lang.cuda import MmaConfig, mma_sync, cp_async, ldmatrix, cp_async_wait_all
from hidet.graph.ops.utils import Task, Operator, Tensor, TensorNode, compute, input_like
from hidet.graph.ops.utils import broadcast_shape, broadcast_shapes, broadcast_indices
from hidet.graph.ops.utils import can_broadcast, schedule_utils
from hidet.utils.py import cdiv, prod
from .attention_mask import AttnMaskAddOp


class AttnTask(Task):
    def __init__(self, name: str, q: TensorNode, k: TensorNode, v: TensorNode, is_causal: bool):
        q_shape = q.const_shape
        k_shape = k.const_shape
        v_shape = v.const_shape
        n_size = q_shape[-2]
        n_kv_size = k_shape[-1]
        d_size = q_shape[-1]
        o_shape = broadcast_shapes([q_shape[:-2], k_shape[:-2], v_shape[:-2]]) + [n_size, d_size]
        o_head, q_head, k_head, v_head = o_shape[:-2], q_shape[:-2], k_shape[:-2], v_shape[:-2]
        qk_head = broadcast_shape(q_head, k_head)

        # ToDo: Add causal mask to compute definition (Will not affect results since schedule template will be used)
        qk = compute(
            name='qk',
            shape=qk_head + [n_size, n_kv_size],
            fcompute=lambda *indices: reduce(
                shape=[d_size],
                fcompute=lambda d: q[broadcast_indices(indices[:-2], q_head, qk_head) + [indices[-2], d]]
                * k[broadcast_indices(indices[:-2], k_head, qk_head) + [d, indices[-1]]],
                reduce_type='sum',
            ),
        )

        qk_shape = qk.const_shape
        axis = len(qk_shape) - 1
        axis_extent = qk_shape[axis]
        reduced_shape = qk_shape[:axis] + qk_shape[axis + 1 :]

        max_value = compute(
            name='max_value',
            shape=reduced_shape,
            fcompute=lambda *indices: reduce(
                shape=[axis_extent], fcompute=lambda a: qk[indices[:axis] + (a,) + indices[axis:]], reduce_type='max'
            ),
        )

        exp_value = compute(
            name='exp_value',
            shape=qk_shape,
            fcompute=lambda *indices: prim.exp(qk[indices] - max_value[indices[:axis] + indices[axis + 1 :]]),
        )

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
        super().__init__(name=name, inputs=[q, k, v], outputs=[o], attributes={'is_causal': is_causal})

    def allow_prologue(self) -> bool:
        return False

    def allow_epilogue(self) -> bool:
        return False

    def implement_cuda(self, working_dir: str) -> Union[List[IRModule], IRModule]:
        return tune.extract_ir_modules(self.cuda_schedule_attn)

    @tune.space(
        2,
        block_i=[128, 64, 256, 512, 32, 16],
        block_j=[128, 64, 256, 512, 32, 16],
        block_k=[8, 16, 32, 64],
        warp_elems_m=[16, 32, 64, 128],
        warp_elems_n=[16, 32, 64, 128],
        warp_elems_k=[8, 16, 32, 64],
        mma_config=[MmaConfig.m16n8k8_f16_f16(), MmaConfig.m16n8k16_f16_f16()],
    )
    @tune.space(
        1,
        block_i=[64],
        block_j=[128],
        block_k=[32],
        warp_elems_m=[16],
        warp_elems_n=[128],
        warp_elems_k=[32],
        mma_config=[MmaConfig.m16n8k8_f16_f16()],
    )
    def cuda_schedule_attn(
        self,
        block_i=128,
        block_j=128,
        block_k=16,
        warp_elems_m=32,
        warp_elems_n=64,
        warp_elems_k=16,
        mma_config=MmaConfig.m16n8k8_f16_f16(),
    ) -> IRModule:
        def calc_swizzle_size(d):
            powers_of_two = [128, 64, 32, 16, 8]
            for n in powers_of_two:
                if d == n:
                    return d, 1
                if d % n == 0:
                    return n, d // n
            return -1, -1

        compute_capability = hidet.cuda.compute_capability()
        compute_capability = compute_capability[0] * 10 + compute_capability[1]
        if compute_capability < 80:
            # hack: sm75 only supports m16n8k8, not m16n8k16
            tune.check(mma_config.k == 8)

        task = self
        is_causal = task.attrs['is_causal']
        node_q, node_k, node_v, node_o = task.inputs[0], task.inputs[1], task.inputs[2], task.outputs[0]
        q_shape: List[int] = list(node_q.const_shape)
        k_shape: List[int] = list(node_k.const_shape)
        v_shape: List[int] = list(node_v.const_shape)
        o_shape: List[int] = list(node_o.const_shape)
        q_head, k_head, v_head, o_head = q_shape[:-2], k_shape[:-2], v_shape[:-2], o_shape[:-2]
        qk_head = broadcast_shape(q_head, k_head)
        bs_qk = prod(qk_head)
        bs = prod(o_head)
        assert bs == bs_qk

        local_layout = DataLayout.local
        row_major = DataLayout.row_major
        n_size = q_shape[-2]
        d_size = q_shape[-1]
        n_kv_size = k_shape[-1]
        dpad_size = 32 * cdiv(d_size, 32)
        dtype = task.inputs[0].ttype.dtype
        dtype_size = dtype.nbytes
        warp_size = 32
        tune.check(d_size % 8 == 0)
        tune.check(d_size <= 160)

        acc_dtype = f16  # must be f16 for now. f32 will fail to compile
        sm_dtype = f32  # currently changing to f16 will not boost performance
        mma_m = mma_config.m
        mma_n = mma_config.n
        mma_k = mma_config.k

        swizzle_unit, swizzle_repeat = calc_swizzle_size(dpad_size)
        tune.check(swizzle_repeat > 0)
        tune.check(block_k == warp_elems_k)
        tune.check(block_i % warp_elems_m == 0)
        tune.check(block_j % warp_elems_n == 0)

        # Number of warps in each dimension. 1, 4, 1
        warp_count_m, warp_count_n, warp_count_k = (
            cdiv(block_i, warp_elems_m),
            cdiv(block_j, warp_elems_n),
            cdiv(block_k, warp_elems_k),
        )
        num_warps = warp_count_m * warp_count_n * warp_count_k
        block_size = num_warps * warp_size
        tune.check(block_size <= 1024)
        # Number of m16n8k16 mma's each warp performs in each dim.
        mmas_per_warp_m, mmas_per_warp_n, mmas_per_warp_k = (
            warp_elems_m // mma_m,
            warp_elems_n // mma_n,
            warp_elems_k // mma_k,
        )

        warp_count_m_o, warp_count_k_o = warp_count_m, 1
        warp_count_n_o = num_warps // (warp_count_m_o * warp_count_k_o)
        block_i_o, block_j_o, block_k_o = block_i, dpad_size, block_k
        tune.check(block_i_o % warp_count_m_o == 0)
        tune.check(block_j_o % warp_count_n_o == 0)
        tune.check(block_k_o % warp_count_k_o == 0)
        tune.check(dpad_size % block_k == 0)
        tune.check(block_j % block_k_o == 0)
        k_tiles = cdiv(dpad_size, block_k)
        k_tiles_o = cdiv(block_j, block_k_o)
        warp_elems_m_o, warp_elems_n_o, warp_elems_k_o = (
            block_i_o // warp_count_m_o,
            block_j_o // warp_count_n_o,
            block_k_o // warp_count_k_o,
        )
        tune.check(warp_elems_m_o % mma_m == 0)
        tune.check(warp_elems_n_o % mma_n == 0)
        tune.check(warp_elems_k_o % mma_k == 0)
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

        n_tiles = cdiv(n_size, block_i)
        i_split = n_tiles
        i_tiles_per_tb = 1
        i_rows_per_tb = i_tiles_per_tb * block_i

        smem_bytes_q = dtype_size * block_i * dpad_size
        # k and v requires double memory for double buffering pipeline
        smem_bytes_k = dtype_size * block_k * block_j * 2
        smem_bytes_v = dtype_size * block_k_o * block_j_o * 2
        smem_bytes_k_v = max(smem_bytes_k, smem_bytes_v)
        smem_bytes_qk = dtype_size * block_i * block_j
        smem_bytes_l = sm_dtype.nbytes * i_rows_per_tb
        smem_bytes_m = sm_dtype.nbytes * i_rows_per_tb
        smem_bytes_lij = sm_dtype.nbytes * block_i
        smem_bytes_mij = sm_dtype.nbytes * block_i

        smem_bytes_offsets = {
            'q': 0,
            'o': 0,
            'k': smem_bytes_q,
            'v': smem_bytes_q,
            'qk': smem_bytes_q + smem_bytes_k_v,
            'l': smem_bytes_q + smem_bytes_k_v + smem_bytes_qk,
            'm': smem_bytes_q + smem_bytes_k_v + smem_bytes_qk + smem_bytes_l,
            'lij': smem_bytes_q + smem_bytes_k_v + smem_bytes_qk + smem_bytes_l + smem_bytes_m,
            'mij': smem_bytes_q + smem_bytes_k_v + smem_bytes_qk + smem_bytes_l + smem_bytes_m + smem_bytes_lij,
        }

        dynamic_smem_bytes = (
            smem_bytes_q
            + smem_bytes_k_v
            + smem_bytes_qk
            + smem_bytes_l
            + smem_bytes_m
            + smem_bytes_lij
            + smem_bytes_mij
        )
        used_smem_bytes_per_block = dynamic_smem_bytes
        smem_limits = {70: 96000, 72: 96000, 75: 64000, 80: 163000, 86: 99000, 87: 163000, 89: 99000, 90: 227000}
        max_smem = 99000 if compute_capability > 90 else smem_limits[compute_capability]
        tune.check(used_smem_bytes_per_block <= max_smem)

        smem_l_type = tensor_type(sm_dtype, shape=[i_rows_per_tb])
        smem_m_type = tensor_type(sm_dtype, shape=[i_rows_per_tb])
        smem_lij_type = tensor_type(sm_dtype, shape=[block_i])
        smem_mij_type = tensor_type(sm_dtype, shape=[block_i])

        smem_q_layout = (
            row_major((1, swizzle_repeat)) * row_major((block_i, swizzle_unit // 8)).swizzle(1) * row_major((1, 8))
        )

        smem_k_layout = row_major((block_k // 8, block_j // 64)) * row_major((8, 8)).swizzle(1) * row_major((1, 8))
        smem_qk_layout = row_major((block_i, block_j // 8)).swizzle(1) * row_major((1, 8))
        if block_j_o % 64 == 0:
            smem_v_layout = (
                row_major((block_k_o // 8, block_j_o // 64)) * row_major((8, 8)).swizzle(1) * row_major((1, 8))
            )
        else:
            smem_v_layout = (
                row_major((1, swizzle_repeat))
                * row_major((block_k_o, swizzle_unit // 8)).swizzle(1)
                * row_major((1, 8))
            )

        smem_q_type = tensor_type('float16', shape=[block_i, dpad_size], layout=smem_q_layout)
        smem_k_type = tensor_type('float16', shape=[block_k, block_j], layout=smem_k_layout)
        smem_k_db_type = tensor_type('float16', shape=[2, block_k, block_j], layout=row_major([2]) + smem_k_layout)
        smem_qk_type = tensor_type('float16', shape=[block_i, block_j], layout=smem_qk_layout)
        smem_v_type = tensor_type('float16', shape=[block_k_o, block_j_o], layout=smem_v_layout)
        smem_v_db_type = tensor_type('float16', shape=[2, block_k_o, block_j_o], layout=row_major([2]) + smem_v_layout)
        regs_o_type = tensor_type(acc_dtype, shape=[mmas_per_warp_m_o, mmas_per_warp_n_o, mma_config.c_elements])

        n_size_per_thread = cdiv(i_rows_per_tb, block_size)
        lm_layout = repeat(n_size_per_thread) * spatial(min(i_rows_per_tb, block_size))

        rows_per_thread_per_mma = 2
        rows_per_thread_mma_o = rows_per_thread_per_mma * mmas_per_warp_m_o
        regs_li_new_layout = row_major((rows_per_thread_mma_o, 1)) * local_layout((mma_m // rows_per_thread_per_mma, 1))
        regs_mi_new_layout = regs_li_new_layout
        regs_exp_mij_layout = regs_li_new_layout

        q_elems_per_thread = block_i * dpad_size // block_size
        q_elems_per_thread = 8 * cdiv(q_elems_per_thread, 8)

        t_per_block_k_8_floor = block_size // (dpad_size // 8)
        if block_i < t_per_block_k_8_floor:
            q_g2s_layout = spatial(block_i, dpad_size // 8)
        else:
            q_g2s_layout = repeat(cdiv(block_i, t_per_block_k_8_floor), 1) * spatial(
                t_per_block_k_8_floor, dpad_size // 8
            )
        k_g2s_layout = repeat(cdiv(block_k, (block_size // (block_j // 8))), 1) * spatial(
            block_size // (block_j // 8), block_j // 8
        )
        if block_k_o < t_per_block_k_8_floor:
            v_g2s_layout = spatial(block_k_o, block_j_o // 8)
        else:
            v_g2s_layout = repeat(cdiv(block_k_o, t_per_block_k_8_floor), 1) * spatial(
                t_per_block_k_8_floor, block_j_o // 8
            )

        q_g2s_layout_sm75, _ = schedule_utils.get_transfer_task_map(
            task_shape=[block_i, dpad_size], num_workers=min(block_i * dpad_size, block_size), ranks=[0, 1]
        )
        k_g2s_layout_sm75, _ = schedule_utils.get_transfer_task_map(
            task_shape=[block_k, block_j], num_workers=min(block_k * block_j, block_size), ranks=[0, 1]
        )
        v_g2s_layout_sm75, _ = schedule_utils.get_transfer_task_map(
            task_shape=[block_k_o, block_j_o], num_workers=min(block_k_o, block_j_o, block_size), ranks=[0, 1]
        )

        with hidet.script_module() as module:
            # --------------- helper functions ---------------------------------------------------------------------
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
            def cp_async_sync():
                if compute_capability >= 80 and (n_kv_size % 8 == 0 or n_size % 8 == 0):
                    cp_async_wait_all()

            @hidet.script
            def init_lm_smem(smem_l: smem_l_type, smem_m: smem_m_type):
                for i in lm_layout.on(threadIdx.x):
                    if i < smem_l_type.shape[0]:
                        smem_l[i] = smem_l_type.dtype.zero
                        smem_m[i] = smem_m_type.dtype.min_value

            @hidet.script
            def copy_k_g2s_sm80(
                k: f16[k_head + [d_size, n_kv_size]], smem_k: smem_k_type, offset_j: i32, offset_k: i32
            ):
                o_head_index = spatial(*o_head).map(blockIdx.y)
                gmem_k = k[broadcast_indices(o_head_index, k_head, o_head)][offset_k:, offset_j:]
                for i, j_seg in k_g2s_layout.on(threadIdx.x):
                    j = j_seg * 8
                    src_size = (
                        0
                        if (offset_k + i >= d_size or offset_j + j >= n_kv_size)
                        else min(n_kv_size - (offset_j + j), 8)
                    )
                    if threadIdx.x < k_g2s_layout.num_workers and i < smem_k_type.shape[0]:
                        cp_async(~smem_k[i, j], ~gmem_k[i, j], cp_size=16, src_size=src_size * 2, cache_level='global')

            @hidet.script
            def copy_v_g2s_sm80(v: f16[v_head + [n_kv_size, d_size]], smem_v: smem_v_type, offset_j: i32):
                o_head_index = spatial(*o_head).map(blockIdx.y)
                gmem_v = v[broadcast_indices(o_head_index, v_head, o_head)][offset_j:, :]
                for i, j_seg in v_g2s_layout.on(threadIdx.x):
                    j = j_seg * 8
                    src_size = 0 if (offset_j + i >= n_kv_size or j >= d_size) else min(d_size - j, 8)
                    if threadIdx.x < v_g2s_layout.num_workers and i < smem_v_type.shape[0]:
                        cp_async(~smem_v[i, j], ~gmem_v[i, j], cp_size=16, src_size=src_size * 2, cache_level='global')

            @hidet.script
            def copy_q_g2s_sm80(q: f16[q_head + [n_size, d_size]], smem_q: smem_q_type, offset_i: i32):
                o_head_index = spatial(*o_head).map(blockIdx.y)
                gmem_q = q[broadcast_indices(o_head_index, q_head, o_head)][offset_i:, :]
                for i, j_seg in q_g2s_layout.on(threadIdx.x):
                    j = j_seg * 8
                    src_size = 0 if (offset_i + i >= n_size or j >= d_size) else min(d_size - j, 8)
                    if threadIdx.x < q_g2s_layout.num_workers and i < smem_q_type.shape[0]:
                        cp_async(~smem_q[i, j], ~gmem_q[i, j], cp_size=16, src_size=src_size * 2, cache_level='global')

            @hidet.script
            def copy_k_g2s_sm75(
                k: f16[k_head + [d_size, n_kv_size]], smem_k: smem_k_type, offset_j: i32, offset_k: i32
            ):
                o_head_index = spatial(*o_head).map(blockIdx.y)
                gmem_k = k[broadcast_indices(o_head_index, k_head, o_head)][offset_k:, offset_j:]
                for i, j in k_g2s_layout_sm75.on(threadIdx.x):
                    if threadIdx.x < k_g2s_layout_sm75.num_workers and i < smem_k_type.shape[0]:
                        if offset_k + i < d_size and offset_j + j < n_kv_size:
                            smem_k[i, j] = gmem_k.read([i, j], protected=False)
                        else:
                            smem_k[i, j] = f16.zero

            @hidet.script
            def copy_v_g2s_sm75(v: f16[v_head + [n_kv_size, d_size]], smem_v: smem_v_type, offset_j: i32):
                o_head_index = spatial(*o_head).map(blockIdx.y)
                gmem_v = v[broadcast_indices(o_head_index, v_head, o_head)][offset_j:, :]
                for i, j in v_g2s_layout_sm75.on(threadIdx.x):
                    if threadIdx.x < v_g2s_layout_sm75.num_workers and i < smem_v_type.shape[0]:
                        if offset_j + i < n_kv_size and j < d_size:
                            smem_v[i, j] = gmem_v.read([i, j], protected=False)
                        else:
                            smem_v[i, j] = f16.zero

            @hidet.script
            def copy_q_g2s_sm75(q: f16[q_head + [n_size, d_size]], smem_q: smem_q_type, offset_i: i32):
                o_head_index = spatial(*o_head).map(blockIdx.y)
                gmem_q = q[broadcast_indices(o_head_index, q_head, o_head)][offset_i:, :]
                for i, j in q_g2s_layout_sm75.on(threadIdx.x):
                    if threadIdx.x < q_g2s_layout_sm75.num_workers and i < smem_q_type.shape[0]:
                        if offset_i + i < n_size and j < d_size:
                            smem_q[i, j] = gmem_q.read([i, j], protected=False)
                        else:
                            smem_q[i, j] = f16.zero

            @hidet.script
            def copy_k_g2s(k: f16[k_head + [d_size, n_kv_size]], smem_k: smem_k_type, offset_j: i32, offset_k: i32):
                if compute_capability >= 80 and n_kv_size % 8 == 0:
                    copy_k_g2s_sm80(k, smem_k, offset_j, offset_k)
                else:
                    copy_k_g2s_sm75(k, smem_k, offset_j, offset_k)

            @hidet.script
            def copy_v_g2s(v: f16[v_head + [n_kv_size, d_size]], smem_v: smem_v_type, offset_j: i32):
                if compute_capability >= 80 and n_kv_size % 8 == 0:
                    copy_v_g2s_sm80(v, smem_v, offset_j)
                else:
                    copy_v_g2s_sm75(v, smem_v, offset_j)

            @hidet.script
            def copy_q_g2s(q: f16[q_head + [n_size, d_size]], smem_q: smem_q_type, offset_i: i32):
                if compute_capability >= 80 and n_size % 8 == 0:
                    copy_q_g2s_sm80(q, smem_q, offset_i)
                else:
                    copy_q_g2s_sm75(q, smem_q, offset_i)

            @hidet.script
            def copy_o_r2g(o: f16[o_head + [n_size, d_size]], regs_o: regs_o_type, offset_i: i32):
                warp_id, lane_id = threadIdx.x / 32, threadIdx.x % 32
                o_head_index = spatial(*o_head).map(blockIdx.y)
                gmem_o = o[o_head_index][offset_i:, :]
                for k_round in range(warp_count_k):
                    for wi, wj, wk in spatial(warp_count_m_o, warp_count_n_o, warp_count_k_o).on(warp_id):
                        if wk == k_round:
                            for mma_i, mma_j in grid(mmas_per_warp_m_o, mmas_per_warp_n_o):
                                p = 0
                                for ti, tj in mma_config.c_store_map.on(lane_id):
                                    delta_m = wi * warp_elems_m_o + mma_i * mma_m + ti
                                    delta_n = wj * warp_elems_n_o + mma_j * mma_n + tj
                                    if delta_m + offset_i < n_size and delta_n < d_size:
                                        gmem_o[delta_m, delta_n] = regs_o[mma_i, mma_j, p]
                                    p += 1

            @hidet.script
            def copy_q_s2r(
                mma_i: int, mma_k0: int, offset_k: int, regs_q: f16[mma_config.a_elements], smem_q: smem_q_type
            ):
                warp_id, lane_id = threadIdx.x / 32, threadIdx.x % 32
                for wi, _, wk in spatial(warp_count_m, warp_count_n, warp_count_k).on(warp_id):
                    p, q = col_spatial(16, 2).map(lane_id)
                    row_addr = ~smem_q[
                        wi * warp_elems_m + mma_i * mma_m + p, offset_k + wk * warp_elems_k + mma_k0 * mma_k + q * 8
                    ]
                    resolve_ldmatrix(regs_q, row_addr, True)

            @hidet.script
            def copy_k_s2r(mma_j: int, k1: int, regs_k: f16[mma_config.b_elements], smem_k: smem_k_type):
                warp_id, lane_id = threadIdx.x / 32, threadIdx.x % 32
                for _, wj, wk in spatial(warp_count_m, warp_count_n, warp_count_k).on(warp_id):
                    p = col_spatial(16, 2).map(lane_id)[0]
                    row_addr = ~smem_k[wk * warp_elems_k + k1 * mma_k + p, wj * warp_elems_n + mma_j * mma_n]
                    resolve_ldmatrix(regs_k, row_addr, False)

            @hidet.script
            def copy_qk_s2r(
                mma_i: int, mma_k0: int, offset_k: int, regs_qk: f16[mma_config.a_elements], smem_qk: smem_qk_type
            ):
                warp_id, lane_id = threadIdx.x / 32, threadIdx.x % 32
                for wi, _, wk in spatial(warp_count_m_o, warp_count_n_o, warp_count_k_o).on(warp_id):
                    if not warp_id >= spatial(warp_count_m_o, warp_count_n_o, warp_count_k_o).num_workers:
                        p, q = col_spatial(16, 2).map(lane_id)
                        row_addr = ~smem_qk[
                            wi * warp_elems_m_o + mma_i * mma_m + p,
                            offset_k + wk * warp_elems_k_o + mma_k0 * mma_k + q * 8,
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
                offset_j: i32,
            ):
                warp_mask = active_mask()
                warp_id, lane_id = threadIdx.x / 32, threadIdx.x % 32
                wi, wj, _ = spatial(warp_count_m, warp_count_n, warp_count_k).map(warp_id)

                # mask out unused values when block_j > (n_kv_size - offset_j)
                for mma_i, mma_j in grid(mmas_per_warp_m, mmas_per_warp_n):
                    p = 0
                    for i, j in mma_config.c_store_map.on(lane_id):
                        delta_n = offset_j + wj * warp_elems_n + mma_j * mma_n + j
                        if delta_n >= n_kv_size:
                            regs_acc[mma_i, mma_j, p] = acc_dtype.min_value
                        p += 1

                # Each thread holds c elements in 2 rows in mma
                rv = register_tensor(acc_dtype, [2])

                # Reduce mij
                rv[0] = acc_dtype.min_value
                rv[1] = acc_dtype.min_value
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
                        if delta_n + offset_j >= n_kv_size:
                            regs_acc[mma_i, mma_j, p] = acc_dtype.zero
                        else:
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

            # -------------- main function ---------------------------------------------------------------
            @hidet.script
            def attn_kernel(
                q: f16[q_head + [n_size, d_size]],
                k: f16[k_head + [d_size, n_kv_size]],
                v: f16[v_head + [n_kv_size, d_size]],
                o: f16[o_head + [n_size, d_size]],
            ):
                attrs.cuda.grid_dim = (i_split, bs)
                attrs.cuda.block_dim = block_size
                attrs.cuda.min_blocks = 1
                attrs.cuda.dynamic_smem_bytes = dynamic_smem_bytes

                offset_i = blockIdx.x * i_rows_per_tb

                smem_q = tensor_pointer('float16', shape=smem_q_type.shape, layout=smem_q_type.layout)
                smem_k = tensor_pointer('float16', shape=smem_k_db_type.shape, layout=smem_k_db_type.layout)
                smem_qk = tensor_pointer('float16', shape=smem_qk_type.shape, layout=smem_qk_type.layout)
                smem_v = tensor_pointer('float16', shape=smem_v_db_type.shape, layout=smem_v_db_type.layout)
                smem_l = tensor_pointer(smem_l_type.dtype, shape=smem_l_type.shape)
                smem_m = tensor_pointer(smem_m_type.dtype, shape=smem_m_type.shape)
                smem_lij = tensor_pointer(smem_lij_type.dtype, shape=smem_lij_type.shape)
                smem_mij = tensor_pointer(smem_mij_type.dtype, shape=smem_mij_type.shape)

                smem_q = dynamic_shared_memory(byte_offset=smem_bytes_offsets['q'], dtype=f16)
                smem_k = dynamic_shared_memory(byte_offset=smem_bytes_offsets['k'], dtype=f16)
                smem_qk = dynamic_shared_memory(byte_offset=smem_bytes_offsets['qk'], dtype=f16)
                smem_v = dynamic_shared_memory(byte_offset=smem_bytes_offsets['v'], dtype=f16)
                smem_l = dynamic_shared_memory(byte_offset=smem_bytes_offsets['l'], dtype=smem_l_type.dtype)
                smem_m = dynamic_shared_memory(byte_offset=smem_bytes_offsets['m'], dtype=smem_m_type.dtype)
                smem_lij = dynamic_shared_memory(byte_offset=smem_bytes_offsets['lij'], dtype=smem_lij_type.dtype)
                smem_mij = dynamic_shared_memory(byte_offset=smem_bytes_offsets['mij'], dtype=smem_mij_type.dtype)

                regs_q = tensor('register', dtype='float16', shape=[2, mmas_per_warp_m, mma_config.a_elements])
                regs_k = tensor('register', dtype='float16', shape=[2, mmas_per_warp_n, mma_config.b_elements])
                regs_acc = tensor(
                    'register', dtype=acc_dtype, shape=[mmas_per_warp_m, mmas_per_warp_n, mma_config.c_elements]
                )
                regs_qk = tensor('register', dtype='float16', shape=[2, mmas_per_warp_m_o, mma_config.a_elements])
                regs_v = tensor('register', dtype='float16', shape=[2, mmas_per_warp_n_o, mma_config.b_elements])
                regs_acc_o = tensor(
                    'register', dtype=acc_dtype, shape=[mmas_per_warp_m_o, mmas_per_warp_n_o, mma_config.c_elements]
                )
                regs_o = tensor('register', dtype=acc_dtype, shape=regs_o_type.shape)
                regs_li_new = tensor('register', dtype=smem_l_type.dtype, layout=regs_li_new_layout)
                regs_mi_new = tensor('register', dtype=smem_m_type.dtype, layout=regs_mi_new_layout)
                regs_exp_mij = tensor('register', dtype=smem_mij_type.dtype, layout=regs_exp_mij_layout)

                init_lm_smem(smem_l, smem_m)
                # Load Qi into Smem, it stays there forever
                copy_q_g2s(q, smem_q, offset_i)

                for a, b, c in grid(mmas_per_warp_m_o, mmas_per_warp_n_o, mma_config.c_elements):
                    regs_acc_o[a, b, c] = acc_dtype.zero
                    regs_o[a, b, c] = acc_dtype.zero

                j_tiles = cdiv(n_kv_size, block_j)
                if is_causal:
                    j_tiles = min(cdiv((blockIdx.x + 1) * block_i, block_j), j_tiles)
                for j in range(j_tiles):
                    offset_j = block_j * j

                    # ----------------------------
                    # Compute QK = Qi * Kj
                    # Init regs_acc to 0
                    for a, b, c in grid(mmas_per_warp_m, mmas_per_warp_n, mma_config.c_elements):
                        regs_acc[a, b, c] = acc_dtype.zero

                    # Copy first tile of k into shared memory
                    copy_k_g2s(k, ~smem_k[0, 0, 0], offset_j, 0)
                    cp_async_sync()
                    syncthreads()

                    for k0 in range(k_tiles):
                        # Load next tile of k
                        copy_k_g2s(k, ~smem_k[(k0 + 1) % 2, 0, 0], offset_j, (k0 + 1) * block_k)
                        for mma_j in range(mmas_per_warp_n):
                            copy_k_s2r(mma_j, 0, ~regs_k[0, mma_j, 0], ~smem_k[k0 % 2, 0, 0])
                        for mma_i in range(mmas_per_warp_m):
                            copy_q_s2r(mma_i, 0, k0 * block_k, ~regs_q[0, mma_i, 0], smem_q)
                        for mma_k in range(mmas_per_warp_k):
                            if mma_k + 1 < mmas_per_warp_k:
                                for mma_j in range(mmas_per_warp_n):
                                    copy_k_s2r(
                                        mma_j, mma_k + 1, ~regs_k[(mma_k + 1) % 2, mma_j, 0], ~smem_k[k0 % 2, 0, 0]
                                    )
                                for mma_i in range(mmas_per_warp_m):
                                    copy_q_s2r(
                                        mma_i, mma_k + 1, k0 * block_k, ~regs_q[(mma_k + 1) % 2, mma_i, 0], smem_q
                                    )
                            for mma_i, mma_j in grid(mmas_per_warp_m, mmas_per_warp_n):
                                warp_mma(
                                    ~regs_q[mma_k % 2, mma_i, 0],
                                    ~regs_k[mma_k % 2, mma_j, 0],
                                    ~regs_acc[mma_i, mma_j, 0],
                                )
                        cp_async_sync()
                        syncthreads()

                    # Preload first tile of v into shared memory
                    copy_v_g2s(v, ~smem_v[0, 0, 0], offset_j)

                    # Apply Causal Masking
                    if is_causal:
                        for mma_i, mma_j in grid(mmas_per_warp_m, mmas_per_warp_n):
                            warp_id, lane_id = threadIdx.x / 32, threadIdx.x % 32
                            wi, wj, wk = spatial(warp_count_m, warp_count_n, warp_count_k).map(warp_id)
                            p = 0
                            for ti, tj in mma_config.c_store_map.on(lane_id):
                                delta_m = offset_i + wi * warp_elems_m + mma_i * mma_m + ti
                                delta_n = offset_j + wj * warp_elems_n + mma_j * mma_n + tj
                                if delta_n > delta_m:
                                    regs_acc[mma_i, mma_j, p] = acc_dtype.min_value
                                p += 1

                    # Iterative softmax, and write result matrix into shared memory
                    qk_softmax_reduce(smem_qk, smem_mij, smem_lij, regs_acc, offset_j)
                    # ----------------------------

                    # ----------------------------
                    # Compute O = QK * V
                    for a, b, c in grid(mmas_per_warp_m_o, mmas_per_warp_n_o, mma_config.c_elements):
                        regs_acc_o[a, b, c] = acc_dtype.zero

                    cp_async_sync()
                    syncthreads()
                    for k1 in range(k_tiles_o):
                        # Load Vj into Smem
                        copy_v_g2s(v, ~smem_v[(k1 + 1) % 2, 0, 0], offset_j + (k1 + 1) * block_k_o)
                        for mma_j in range(mmas_per_warp_n_o):
                            copy_v_s2r(mma_j, 0, ~regs_v[0, mma_j, 0], ~smem_v[k1 % 2, 0, 0])
                        for mma_i in range(mmas_per_warp_m_o):
                            copy_qk_s2r(mma_i, 0, k1 * block_k_o, ~regs_qk[0, mma_i, 0], smem_qk)
                        for mma_k in range(mmas_per_warp_k_o):
                            if mma_k + 1 < mmas_per_warp_k:
                                for mma_j in range(mmas_per_warp_n_o):
                                    copy_v_s2r(
                                        mma_j, mma_k + 1, ~regs_v[(mma_k + 1) % 2, mma_j, 0], ~smem_v[k1 % 2, 0, 0]
                                    )
                                for mma_i in range(mmas_per_warp_m_o):
                                    copy_qk_s2r(
                                        mma_i, mma_k + 1, k1 * block_k_o, ~regs_qk[(mma_k + 1) % 2, mma_i, 0], smem_qk
                                    )
                            for mma_i, mma_j in grid(mmas_per_warp_m_o, mmas_per_warp_n_o):
                                warp_mma(
                                    ~regs_qk[mma_k % 2, mma_i, 0],
                                    ~regs_v[mma_k % 2, mma_j, 0],
                                    ~regs_acc_o[mma_i, mma_j, 0],
                                )
                        cp_async_sync()
                        syncthreads()
                    # ----------------------------

                    # ----------------------------
                    # Compute final O based on previous and current softmax
                    offset_lm_i = 0
                    warp_id, lane_id = threadIdx.x / 32, threadIdx.x % 32
                    for k_round in range(warp_count_k_o):
                        for wi, _, wk in spatial(warp_count_m_o, warp_count_n_o, warp_count_k_o).on(warp_id):
                            if wk == k_round:
                                for mma_i, mma_j in grid(mmas_per_warp_m_o, 1):
                                    c_store_map = repeat(2, 1) * spatial(8, 4)
                                    for ti, _ in c_store_map.on(lane_id):
                                        delta_m = wi * warp_elems_m_o + mma_i * mma_m + ti
                                        delta_m_reg = delta_m % (mma_m * mmas_per_warp_m_o)
                                        mi = smem_m[offset_lm_i + delta_m]
                                        mij = smem_mij[delta_m]
                                        li = smem_l[offset_lm_i + delta_m]
                                        lij = smem_lij[delta_m]
                                        syncthreads()
                                        regs_mi_new[delta_m_reg, 0] = prim.max(mi, mij)
                                        smem_m[offset_lm_i + delta_m] = regs_mi_new[delta_m_reg, 0]
                                        exp_mi = prim.exp(mi - regs_mi_new[delta_m_reg, 0])
                                        exp_mij = prim.exp(mij - regs_mi_new[delta_m_reg, 0])
                                        # reuse regs_mi_new
                                        regs_mi_new[delta_m_reg, 0] = exp_mi * li
                                        regs_li_new[delta_m_reg, 0] = exp_mi * li + exp_mij * lij
                                        smem_l[offset_lm_i + delta_m] = regs_li_new[delta_m_reg, 0]
                                        regs_exp_mij[delta_m_reg, 0] = exp_mij
                                        syncthreads()

                    for k_round in range(warp_count_k_o):
                        for wi, _, wk in spatial(warp_count_m_o, warp_count_n_o, warp_count_k_o).on(warp_id):
                            if wk == k_round:
                                for mma_i, mma_j in grid(mmas_per_warp_m_o, mmas_per_warp_n_o):
                                    p = 0
                                    for ti, _ in mma_config.c_store_map.on(lane_id):
                                        delta_m = wi * warp_elems_m_o + mma_i * mma_m + ti
                                        delta_m_reg = delta_m % (mma_m * mmas_per_warp_m_o)
                                        regs_o[mma_i, mma_j, p] = (
                                            regs_mi_new[delta_m_reg, 0] * regs_o[mma_i, mma_j, p]
                                            + regs_exp_mij[delta_m_reg, 0] * regs_acc_o[mma_i, mma_j, p]
                                        ) / regs_li_new[delta_m_reg, 0]
                                        p += 1
                    syncthreads()
                    # ----------------------------
                # } end of main k tile loop

                copy_o_r2g(o, regs_o, offset_i)
                syncthreads()

        ir_module = module.ir_module()
        return ir_module


class AttnOp(Operator):
    def __init__(self, q: Tensor, k: Tensor, v: Tensor, is_causal: bool = False):
        super().__init__(
            inputs=[q, k, v],
            task=AttnTask('attn', input_like(q, 'q'), input_like(k, 'k'), input_like(v, 'v'), is_causal),
            attributes={'is_causal': is_causal},
        )


def attention(q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None, is_causal: bool = False) -> Tensor:
    if mask is not None and is_causal is True:
        raise ValueError("mask and is_causal cannot be set at the same time")

    if not q.dtype == k.dtype == v.dtype == f16:
        raise ValueError("Attention only supports float16 inputs")

    if not (
        len(q.shape) == len(k.shape) == len(v.shape)
        and k.shape[-1] == v.shape[-2]
        and q.shape[-1] == k.shape[-2] == v.shape[-1]
    ):
        raise ValueError(
            'Attention expect tensor Q[..., S_q, D], K[..., D, S_kv], V[..., S_kv, D]'
            + ', got Q {}, K {}, V {}'.format(q.shape, k.shape, v.shape)
        )

    if q.shape[-1] > 160:
        raise ValueError('Attention only supports head dim <= 160, got {}'.format(q.shape[-1]))

    if mask is None:
        return AttnOp(q, k, v, is_causal).get_output(0)

    mask_shape = mask.shape
    seq_len_q = q.shape[-2]
    seq_len_kv = v.shape[-2]
    q_head, k_head = (q.shape[:-2], k.shape[:-2])
    qk_head = broadcast_shape(q_head, k_head)
    qk_shape = qk_head + [seq_len_q, seq_len_kv]
    if not can_broadcast(mask_shape, qk_shape):
        raise ValueError("Invalid mask dimension: {}".format(mask_shape))

    return AttnMaskAddOp(q, k, v, mask).get_output(0)
