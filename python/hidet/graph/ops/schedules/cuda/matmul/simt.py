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
pseudo code of matmul with double buffering
=========
assume block_k % task_k == 0 and warp_k % block_k == 0
gmem[0] -> smem[0]
sync
smem[0, 0] -> regs[0]
sync
for k0 in range(task_k / block_k - 1):
    for k1 in range(block_k / warp_k):
        if k1 == 0:
            smem[k0 % 2, k1+1] -> regs[(k1 + 1) % 2]
            gmem[k0 + 1] -> smem[(k0 + 1) % 2]
            regs[k1 % 2] -> acc regs
        elif 0 < k1 < block_k / warp_k - 1:
            smem[k0 % 2, k1+1] -> regs[(k1 + 1) % 2]
            regs[k1 % 2] -> acc regs
        else k1 == block_k / warp_k - 1:
            sync
            smem[(k0 + 1) % 2, 0] -> regs[(k1 + 1) % 2]
            regs[k1 % 2] -> acc regs
k0 = task_k / block_k - 1
for k1 in range(block_k / warp_k):
    if k1 == 0:
        smem[k0 % 2, k1+1] -> regs[(k1 + 1) % 2]
        regs[k1 % 2] -> acc regs
    elif 0 < k1 < block_k / warp_k - 1:
        smem[k0 % 2, k1+1] -> regs[(k1 + 1) % 2]
        regs[k1 % 2] -> acc regs
    else k1 == block_k / warp_k - 1:
        regs[k1 % 2] -> acc regs
sync
write back
"""
from typing import List, Tuple, Union, Optional

import os

import hidet.cuda
from hidet import option
from hidet.ir.builders import FunctionBuilder, StmtBuilder
from hidet.ir.expr import Var, LogicalAnd, Equal, Cast, if_then_else, convert, Expr
from hidet.ir.func import IRModule
from hidet.ir.functors import simplify_to_int
from hidet.ir.mapping import TaskMapping
from hidet.ir.layout import DataLayout, StridesLayout
from hidet.ir.primitives import syncthreads, thread_idx, block_idx
from hidet.ir.stmt import BufferStoreStmt, IfStmt, DeclareStmt, DeclareScope
from hidet.ir.type import data_type, tensor_type, PointerType, tensor_pointer_type
from hidet.graph.ops.definitions.matmul import BatchMatmulTask
from hidet.graph.ops.schedules.resolve import resolve_ir_modules
from hidet.graph.ops.schedules.common import params_from_task, Schedule, NotSupportedError
from hidet.transforms.tools import fuse_and_pack


class MatmulSchedule(Schedule):
    def __init__(
        self,
        block_warps_k=8,
        warp_k=1,
        block_warps=(4, 2),
        warp_outer=(2, 2),
        atom_layout=TaskMapping.row_major([4, 8]),
        atom_layout_name='row_4x8',
        warp_inner=(4, 4),
        dtype='float32',
    ):
        self.block_warps_k = block_warps_k
        self.warp_k = warp_k
        self.block_warps = block_warps
        self.warp_outer = warp_outer
        self.atom_layout = atom_layout
        self.warp_inner = warp_inner
        self.atom_layout_name = atom_layout_name

        # sanity check
        row_major = TaskMapping.row_major
        full_layout = TaskMapping.full_layout
        warp_outer_layout = full_layout(warp_outer)
        warp_inner_layout = full_layout(warp_inner)
        warp_layout = warp_outer_layout * atom_layout * warp_inner_layout
        block_warps_layout = row_major(block_warps)
        block_layout = block_warps_layout * warp_layout
        block_k = block_warps_k * warp_k
        atom_shape = atom_layout.task_shape
        block_shape = block_layout.task_shape
        warp_size = 32
        block_size = block_layout.num_workers
        self.check(
            atom_layout.num_workers == 32,
            "atom layout should have exactly 32 workers, corresponding to 32 threads in a warp",
        )
        self.check(block_warps_k % 2 == 0, "double buffering requires that block_k/warp_k is divisible by 2")
        if block_k <= warp_size:
            self.check(
                warp_size % block_k == 0,
                f"transfer from gmem to smem requires block_k ({block_k}) " f"is divisible by warp_size ({warp_size})",
            )
            self.check(
                block_shape[0] % (block_size // block_k) == 0 and block_shape[1] % (block_size // block_k) == 0,
                f"transfer of matrix A/B from gmem to regs requirement. "
                f"block_shape ({block_shape}) block_size ({block_size}) block_k ({block_k}) "
                f"block_size / block_k ({block_size / block_k})",
            )
        else:
            self.check(
                block_k % warp_size == 0, "transfer from gmem to smem requires warp_size is divisible by block_k"
            )
            raise NotSupportedError(self, "Will support later")

        # derived data layouts
        local_layout = DataLayout.local
        row_major = DataLayout.row_major
        col_major = DataLayout.column_major
        self.regs_a_layout = (
            local_layout((block_warps[0], 1))
            * col_major((warp_outer[0], warp_k))
            * local_layout((atom_shape[0], 1))
            * row_major((warp_inner[0], 1))
        )
        self.regs_b_layout = (
            local_layout((1, block_warps[1]))
            * row_major((warp_k, warp_outer[1]))
            * local_layout((1, atom_shape[1]))
            * row_major((1, warp_inner[1]))
        )
        self.regs_c_layout = (
            local_layout(block_warps) * row_major(warp_outer) * local_layout(atom_shape) * row_major(warp_inner)
        )
        if block_k <= warp_size:
            self.regs_a_ldg_layout = local_layout((block_size // block_k, block_k)) * row_major(
                (block_shape[0] // (block_size // block_k), 1)
            )
            self.regs_b_ldg_layout = row_major((1, block_shape[1] // (block_size // block_k))) * local_layout(
                (block_k, block_size // block_k)
            )
        else:
            raise NotSupportedError(self)
        reserved_regs = 48  # number of reserved registers for intermediate results
        used_num_regs_per_thread = (
            self.regs_a_layout.size
            + self.regs_b_layout.size
            + self.regs_c_layout.size
            + self.regs_a_ldg_layout.size
            + self.regs_b_ldg_layout.size
            + reserved_regs
        )
        # the number of registers allocated to each thread is a multiple of 8.
        used_num_regs_per_thread = (used_num_regs_per_thread + 7) // 8 * 8
        resident_blocks = hidet.cuda.properties().regsPerMultiprocessor // (used_num_regs_per_thread * block_size)

        max_smem_bytes_per_block = (
            min(
                # hidet.cuda.properties().sharedMemPerMultiprocessor // resident_blocks,
                48 * 1024 // resident_blocks,
                hidet.cuda.properties().sharedMemPerBlock,
            )
            // 128
            * 128
        )

        # derived task layouts
        row_major = TaskMapping.row_major
        full_layout = TaskMapping.full_layout
        self.block_warps_layout = block_warps_layout
        self.warp_layout = warp_layout
        self.block_layout = block_layout
        if block_k <= warp_size:
            lines = block_size // block_k
            self.a_g2s_layout = row_major([lines, block_k]) * full_layout([block_shape[0] // lines, 1])
            self.b_g2s_layout = full_layout([1, block_shape[1] // lines]) * row_major([block_k, lines])
        else:
            raise NotSupportedError(self)
        self.a_s2r_layout = (
            self.block_warps_layout
            * full_layout([warp_outer[0], warp_k])
            * atom_layout
            * full_layout([warp_inner[0], warp_k])
        ).projection({1: 0})
        self.b_s2r_layout = (
            self.block_warps_layout
            * full_layout([warp_k, warp_outer[1]])
            * atom_layout
            * full_layout([warp_k, warp_inner[1]])
        ).projection({0: 0})

        # derived constants
        used_smem_bytes_per_block = (block_shape[0] + block_shape[1]) * block_k * 2 * data_type(dtype).nbytes
        self.check(
            used_smem_bytes_per_block <= max_smem_bytes_per_block,
            f"Used shared memory ({used_smem_bytes_per_block} bytes) "
            f"exceeded the maximum ({max_smem_bytes_per_block} bytes)",
        )
        self.block_size = block_size
        self.block_shape = block_layout.task_shape
        self.block_k = block_k
        self.warp_shape = warp_layout.task_shape
        self.warp_k = warp_k
        self.used_num_regs_per_thread = used_num_regs_per_thread
        self.used_smem_bytes_per_block = used_smem_bytes_per_block
        # we muse use dynamic shared memory when we use more than 48 KiBytes shared memory
        # see Appendix 'Compute Capability' in
        # CUDA C Programming Guide <https://docs.nvidia.com/cuda/pdf/CUDA_C_Programming_Guide.pdf>
        self.use_dynamic_smem = used_smem_bytes_per_block > 48 * 1024
        self.min_thread_blocks = resident_blocks

        self.check(used_num_regs_per_thread <= 255, f'register used {used_num_regs_per_thread} exceeds maximum {255}')
        self.check(
            used_num_regs_per_thread * block_size <= hidet.cuda.properties().regsPerBlock,
            f'echo block can only have {hidet.cuda.properties().regsPerBlock} registers, '
            f'but this schedule requires {used_num_regs_per_thread * block_size} registers',
        )

    def keys(self) -> List[Tuple[str, Union[int, float, str]]]:
        return [
            ('bwx', self.block_warps[0]),
            ('bwy', self.block_warps[1]),
            ('wox', self.warp_outer[0]),
            ('woy', self.warp_outer[1]),
            ('atom', self.atom_layout_name),
            ('wix', self.warp_inner[0]),
            ('wiy', self.warp_inner[1]),
            ('bk', self.block_k),
            ('wk', self.warp_k),
            ('mtb', self.min_thread_blocks),
        ]

    def derived_keys(self) -> List[Tuple[str, Union[int, float, str]]]:
        return [
            ('bx', self.block_shape[0]),
            ('by', self.block_shape[1]),
            ('regs', self.used_num_regs_per_thread),
            ('smem', self.used_smem_bytes_per_block),
        ]

    def check(self, cond, msg: str = ""):
        if not cond:
            raise NotSupportedError(self, msg)

    @staticmethod
    def schedules(space_level: int = 0):
        settings = []
        if space_level == 0:
            settings.append(MatmulSchedule())
        elif space_level == 1:
            for inner_m, inner_n in [[4, 4], [4, 8], [8, 4]]:
                for outer_m, outer_n in [[1, 1], [1, 2], [2, 1], [2, 2]]:
                    for block_warps_k, warp_k in [[8, 1]]:
                        for block_warps_m, block_warps_n in [[1, 1], [1, 2], [2, 2], [2, 4]]:
                            for name, atom_layout in [('row_4x8', TaskMapping.row_major((4, 8)))]:
                                try:
                                    settings.append(
                                        MatmulSchedule(
                                            block_warps_k=block_warps_k,
                                            warp_k=warp_k,
                                            block_warps=[block_warps_m, block_warps_n],
                                            warp_outer=[outer_m, outer_n],
                                            atom_layout=atom_layout,
                                            atom_layout_name=name,
                                            warp_inner=[inner_m, inner_n],
                                        )
                                    )
                                except NotSupportedError:
                                    pass
        elif space_level == 2:
            grid = TaskMapping.row_major
            for inner_m, inner_n in [[4, 4]]:
                for outer_m, outer_n in [[1, 1], [1, 2], [2, 1], [2, 2], [1, 3], [3, 1], [2, 3], [3, 2], [3, 3]]:
                    for block_warps_k, warp_k in [[4, 1], [8, 1]]:
                        for block_warps_m, block_warps_n in [[1, 1], [1, 2], [2, 1], [2, 2], [2, 4], [4, 2]]:
                            for name, atom_layout in [
                                ('row_4x8', grid((4, 8))),
                                ('custom_4x8', grid((2, 1)) * grid((1, 8)) * grid((2, 1))),
                                ('row_2x16', grid((2, 16))),
                                ('row_16x2', grid((16, 2))),
                                ('row_1x32', grid((1, 32))),
                                ('row_32x1', grid((32, 1))),
                            ]:
                                try:
                                    settings.append(
                                        MatmulSchedule(
                                            block_warps_k=block_warps_k,
                                            warp_k=warp_k,
                                            block_warps=[block_warps_m, block_warps_n],
                                            warp_outer=[outer_m, outer_n],
                                            atom_layout=atom_layout,
                                            atom_layout_name=name,
                                            warp_inner=[inner_m, inner_n],
                                        )
                                    )
                                except NotSupportedError:
                                    pass
        else:
            raise NotImplementedError()
        return settings


def batched_matmul_cuda_schedule_simt(task: BatchMatmulTask, working_dir: str) -> IRModule:
    all_schedules = MatmulSchedule.schedules(space_level=option.get_option('search_space'))
    resolve_out_dir = os.path.join(working_dir, './resolve')
    ir_modules = []
    for schedule in all_schedules:
        ir_modules.append(batched_matmul_cuda_with_given_schedule(task, schedule))
    return resolve_ir_modules(
        ir_modules=ir_modules,
        schedules=all_schedules,
        func_name=task.name,
        target_device='cuda',
        output_dir=resolve_out_dir,
        parallel=True,
        verbose=True,
    )


def batched_matmul_cuda_with_given_schedule(task: BatchMatmulTask, schedule: MatmulSchedule) -> IRModule:
    sch = schedule

    a_dtype = task.inputs[0].ttype.dtype
    b_dtype = task.inputs[1].ttype.dtype
    c_dtype = task.outputs[0].ttype.dtype

    batch_size = task.batch_size
    m_size, k_size, n_size = task.m_size, task.k_size, task.n_size

    m_tile_size, n_tile_size = sch.block_shape
    m_tiles = (m_size + m_tile_size - 1) // m_tile_size
    n_tiles = (n_size + n_tile_size - 1) // n_tile_size
    grid_blocks_layout: TaskMapping = TaskMapping.row_major([m_tiles, n_tiles])

    # define function
    with FunctionBuilder(
        name=task.name + '_grid',
        kind='cuda_kernel',
        grid_dim=(grid_blocks_layout.num_workers, batch_size),
        block_dim=sch.block_size,
        dynamic_smem_bytes=sch.used_smem_bytes_per_block if sch.use_dynamic_smem else 0,
        min_blocks=sch.min_thread_blocks,
        label=str(sch),
    ) as fb:
        sb = StmtBuilder()

        # declare params
        params = params_from_task(task)
        gmem_a, gmem_b, gmem_c = params
        fb.extend_params(params)

        # declare local variables
        smem_a = Var(
            'smem_a',
            tensor_pointer_type(
                a_dtype, layout=StridesLayout.from_shape([2, sch.block_shape[0], sch.block_k], perm=[0, 2, 1])
            ),
        )
        smem_b = Var(
            'smem_b',
            tensor_pointer_type(
                b_dtype, layout=StridesLayout.from_shape([2, sch.block_k, sch.block_shape[1]], perm=[0, 1, 2])
            ),
        )
        if sch.use_dynamic_smem:
            # 'extern __shared__ uint8_t smem_storage[];' in c code
            smem_storage = Var(
                'smem_storage',
                PointerType(base_type=data_type('uint8'), specifiers=['extern', '__shared__'], use_bracket=True),
            )
            sb += DeclareStmt(smem_storage)
        else:
            smem_storage = Var('smem_storage', tensor_type(dtype='uint8', shape=[sch.used_smem_bytes_per_block]))
            sb += DeclareStmt(smem_storage, scope=DeclareScope.Shared)
        smem_a_bytes = simplify_to_int(smem_a.type.tensor_type.storage_bytes())
        sb += DeclareStmt(smem_a, init=Cast(~smem_storage[0], PointerType(a_dtype)), scope=DeclareScope.Shared)
        sb += DeclareStmt(
            smem_b, init=Cast(~(smem_storage[smem_a_bytes]), PointerType(b_dtype)), scope=DeclareScope.Shared
        )

        # declare a, b, c registers
        regs_a = Var('regs_A', tensor_type(a_dtype, layout=[2] + schedule.regs_a_layout))
        regs_b = Var('regs_B', tensor_type(b_dtype, layout=[2] + schedule.regs_b_layout))
        regs_c = Var('regs_C', tensor_type(c_dtype, layout=schedule.regs_c_layout))
        regs_a_ldg = Var('regs_A_ldg', tensor_type(dtype=a_dtype, layout=schedule.regs_a_ldg_layout))
        regs_b_ldg = Var('regs_B_ldg', tensor_type(dtype=b_dtype, layout=schedule.regs_b_ldg_layout))
        sb += DeclareStmt(regs_a)
        sb += DeclareStmt(regs_b)
        sb += DeclareStmt(regs_c)
        sb += DeclareStmt(regs_a_ldg)
        sb += DeclareStmt(regs_b_ldg)

        a_default_value = convert(0.0, a_dtype)
        b_default_value = convert(0.0, b_dtype)
        acc_default_value = convert(0.0, c_dtype)

        with sb.lets(['bi', 'bj'], grid_blocks_layout(block_idx())[0]) as (bi, bj):
            block_k_tiles = (k_size + sch.block_k - 1) // sch.block_k
            first_k_tile = k_size - (block_k_tiles - 1) * sch.block_k
            block_offset = [idx * dim for idx, dim in zip([bi, bj], sch.block_shape)]
            # transfer first tile
            sb += copy(
                gmem_a[block_idx('y'), block_offset[0] :, :],
                regs_a_ldg,
                schedule.a_g2s_layout,
                src_predicate=lambda i, k: LogicalAnd.join(block_offset[0] + i < m_size, k < first_k_tile),
                default_value=a_default_value,
            )
            sb += copy(regs_a_ldg, smem_a[0], layout=schedule.a_g2s_layout)
            sb += copy(
                gmem_b[block_idx('y'), :, block_offset[1] :],
                regs_b_ldg,
                schedule.b_g2s_layout,
                src_predicate=lambda k, j: LogicalAnd.join(k < first_k_tile, block_offset[1] + j < n_size),
                default_value=b_default_value,
            )
            sb += copy(regs_b_ldg, smem_b[0], layout=schedule.b_g2s_layout)
            sb += syncthreads()
            sb += copy(smem_a[0], regs_a[0], schedule.a_s2r_layout)
            sb += copy(smem_b[0], regs_b[0], schedule.b_s2r_layout)
            sb += syncthreads()
            # init regs c
            sb += init(regs_c, acc_default_value, schedule.block_layout)
            with sb.for_loop('k0', block_k_tiles - 1) as k0:
                block_offset_k = k0 * sch.block_k + first_k_tile
                with sb.for_loop('k1', sch.block_warps_k) as k1:
                    with sb.if_then(Equal(k1, sch.block_warps_k - 1)):
                        sb += copy(regs_a_ldg, smem_a[(k0 + 1) % 2], schedule.a_g2s_layout)
                        sb += copy(regs_b_ldg, smem_b[(k0 + 1) % 2], schedule.b_g2s_layout)
                        sb += syncthreads()
                        sb += copy(smem_a[(k0 + 1) % 2], regs_a[(k1 + 1) % 2], schedule.a_s2r_layout)
                        sb += copy(smem_b[(k0 + 1) % 2], regs_b[(k1 + 1) % 2], schedule.b_s2r_layout)
                    with sb.otherwise():
                        sb += copy(smem_a[k0 % 2, :, k1 + 1 :], regs_a[(k1 + 1) % 2], schedule.a_s2r_layout)
                        sb += copy(smem_b[k0 % 2, k1 + 1 :, :], regs_b[(k1 + 1) % 2], schedule.b_s2r_layout)
                    with sb.if_then(Equal(k1, 0)):
                        sb += copy(
                            gmem_a[block_idx('y'), block_offset[0] :, block_offset_k:],
                            regs_a_ldg,
                            layout=schedule.a_g2s_layout,
                            src_predicate=lambda i, _: block_offset[0] + i < m_size,
                            default_value=a_default_value,
                        )
                        sb += copy(
                            gmem_b[block_idx('y'), block_offset_k:, block_offset[1] :],
                            regs_b_ldg,
                            layout=schedule.b_g2s_layout,
                            src_predicate=lambda _, j: block_offset[1] + j < n_size,
                            default_value=b_default_value,
                        )
                    sb += mma(regs_a[k1 % 2], regs_b[k1 % 2], regs_c, schedule)
            with sb.let('block_k_tile', block_k_tiles - 1) as k0:
                with sb.for_loop('warp_k_tile', sch.block_warps_k) as k1:
                    with sb.if_then(k1 < sch.block_warps_k - 1):
                        sb += copy(smem_a[k0 % 2, :, k1 + 1 :], regs_a[(k1 + 1) % 2], schedule.a_s2r_layout)
                        sb += copy(smem_b[k0 % 2, k1 + 1 :, :], regs_b[(k1 + 1) % 2], schedule.b_s2r_layout)
                    sb += mma(regs_a[k1 % 2], regs_b[k1 % 2], regs_c, schedule)
            sb += copy(
                src=regs_c,
                dst=gmem_c[block_idx('y'), block_offset[0] :, block_offset[1] :],
                layout=schedule.block_layout,
                dst_predicate=lambda i, j: LogicalAnd(block_offset[0] + i < m_size, block_offset[1] + j < n_size),
            )
        # set body
        fb.set_body(sb.finish())

    func = fb.get()
    ir_module = IRModule(funcs={func.name: func}, task=task)
    return fuse_and_pack(ir_module, func, task)


def init(dst, init_value, layout):
    sb = StmtBuilder()
    for indices in layout(thread_idx()):
        sb += BufferStoreStmt(dst, indices, init_value)
    return sb.finish()


def copy(src, dst, layout, src_predicate=None, dst_predicate=None, default_value: Optional[Union[Expr, float]] = 0.0):
    sb = StmtBuilder()
    for indices in layout(thread_idx()):
        value = src[indices]
        if src_predicate:
            value = if_then_else(src_predicate(*indices), value, default_value)
        stmt = BufferStoreStmt(dst, indices, value)
        if dst_predicate:
            stmt = IfStmt(dst_predicate(*indices), stmt)
        sb += stmt
    return sb.finish()


def mma(a, b, c, schedule):
    layout = schedule.block_layout
    sb = StmtBuilder()
    for i, j in layout(thread_idx()):
        for k in range(schedule.warp_k):
            sb += BufferStoreStmt(c, [i, j], c[i, j] + a[i, k] * b[k, j])
    return sb.finish()
