import contextlib
import itertools
from typing import Mapping, List, Any, Tuple, Union

import numpy as np

from hidet.backend import batch_build, BuildInstance
from hidet.implement.implementer import Implementer, register_impl, NotSupportedError, Schedule
from hidet.ir.builders import FunctionBuilder, StmtBuilder
from hidet.ir.dialects.compute import TensorInput, TensorCompute, ReduceCompute
from hidet.ir.dialects.lowlevel import TensorPointerType, Cast, PointerType
from hidet.ir.dialects.pattern import TaskPattern, any_const_int
from hidet.ir.expr import var, Var, And, Equal, if_then_else
from hidet.ir.func import IRModule
from hidet.ir.functors import simplify_to_int
from hidet.ir.layout import TaskLayout, row_major_layout, DataLayout, StridesLayout
from hidet.ir.node import Node
from hidet.ir.primitives import syncthreads, thread_idx, block_idx
from hidet.ir.stmt import AssignStmt, BufferStoreStmt, IfStmt
from hidet.ir.task import Task, Grid
from hidet.ir.type import scalar_type, TensorType, Scope
from hidet.utils import Timer, cuda, factor, prod

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


class CustomTaskLayout(TaskLayout):
    def __init__(self):
        super().__init__(num_workers=32, task_shape=(4, 8), worker2task=self._work2task)

    def _work2task(self, w):
        return [(w // 16 * 2 + w % 2, w // 2 % 8)]


class MatmulSchedule(Schedule):
    def __init__(self,
                 block_warps_k=8,
                 warp_k=1,
                 block_warps=(4, 2),
                 warp_outer=(2, 2),
                 atom_layout=CustomTaskLayout(),
                 atom_layout_name='custom_4x8',
                 warp_inner=(4, 4)):
        self.block_warps_k = block_warps_k
        self.warp_k = warp_k
        self.block_warps = block_warps
        self.warp_outer = warp_outer
        self.atom_layout = atom_layout
        self.warp_inner = warp_inner
        self.atom_layout_name = atom_layout_name

        # sanity check
        row_major = TaskLayout.row_major
        full_layout = TaskLayout.full_layout
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
        self.check(atom_layout.num_workers == 32, "atom layout should have exactly 32 workers, corresponding to 32 threads in a warp")
        self.check(block_warps_k % 2 == 0, "double buffering requires that block_k/warp_k is divisible by 2")
        if block_k <= warp_size:
            self.check(warp_size % block_k == 0, f"transfer from gmem to smem requires block_k ({block_k}) is divisible by warp_size ({warp_size})")
            # todo: consider removing the following two constraints by adding bound-checking in source-template
            self.check(block_shape[0] % (block_size // block_k) == 0 and block_shape[1] % (block_size // block_k) == 0,
                       f"transfer of matrix A/B from gmem to regs requirement. block_shape ({block_shape}) block_size ({block_size}) block_k ({block_k}) block_size / block_k ({block_size / block_k})")
        else:
            self.check(block_k % warp_size == 0, "transfer from gmem to smem requires warp_size is divisible by block_k")
            raise NotSupportedError("Will support later")

        # derived data layouts
        local_layout = DataLayout.local
        row_major = DataLayout.row_major
        col_major = DataLayout.column_major
        self.regs_a_layout = local_layout((block_warps[0], 1)) * col_major((warp_outer[0], warp_k)) * local_layout((atom_shape[0], 1)) * row_major((warp_inner[0], 1))
        self.regs_b_layout = local_layout((1, block_warps[1])) * row_major((warp_k, warp_outer[1])) * local_layout((1, atom_shape[1])) * row_major((1, warp_inner[1]))
        self.regs_c_layout = local_layout(block_warps) * row_major(warp_outer) * local_layout(atom_shape) * row_major(warp_inner)
        if block_k <= warp_size:
            self.regs_a_ldg_layout = local_layout((block_size // block_k, block_k)) * row_major((block_shape[0] // (block_size // block_k), 1))
            self.regs_b_ldg_layout = row_major((1, block_shape[1] // (block_size // block_k))) * local_layout((block_k, block_size // block_k))
        else:
            raise NotSupportedError()
        reserved_regs = 16
        used_num_regs_per_thread = self.regs_a_layout.size + self.regs_b_layout.size + self.regs_c_layout.size + self.regs_a_ldg_layout.size + self.regs_b_ldg_layout.size + reserved_regs
        used_num_regs_per_thread = (used_num_regs_per_thread + 7) // 8 * 8  # the number of registers allocated to each thread is a multiple of 8.
        self.check(used_num_regs_per_thread <= cuda.max_num_regs_per_thread(),
                   f'register used {used_num_regs_per_thread} exceeds maximum {cuda.max_num_regs_per_thread()}')
        self.check(used_num_regs_per_thread * block_size <= cuda.max_num_regs_per_block(),
                   f'echo block can only have {cuda.max_num_regs_per_block()} registers, but this schedule requires {used_num_regs_per_thread * block_size} registers')
        resident_blocks = cuda.max_num_regs_per_sm() // (used_num_regs_per_thread * block_size)

        max_smem_bytes_per_block = min(cuda.max_smem_bytes_per_sm() // resident_blocks, cuda.max_smem_bytes_per_block()) // 128 * 128

        # derived task layouts
        row_major = TaskLayout.row_major
        full_layout = TaskLayout.full_layout
        self.block_warps_layout = block_warps_layout
        self.warp_layout = warp_layout
        self.block_layout = block_layout
        if block_k <= warp_size:
            lines = block_size // block_k
            self.a_g2s_layout = row_major([lines, block_k]) * full_layout([block_shape[0] // lines, 1])
            self.b_g2s_layout = full_layout([1, block_shape[1] // lines]) * row_major([block_k, lines])
        else:
            raise NotSupportedError()
        self.a_s2r_layout = (self.block_warps_layout * full_layout([warp_outer[0], warp_k]) * atom_layout * full_layout([warp_inner[0], warp_k])).projection({1: 0})
        self.b_s2r_layout = (self.block_warps_layout * full_layout([warp_k, warp_outer[1]]) * atom_layout * full_layout([warp_k, warp_inner[1]])).projection({0: 0})

        pairs = []
        for a, b in itertools.product(factor(warp_outer[0]), factor(warp_outer[1])):
            used_smem_bytes = prod((block_warps_layout * full_layout([a, b]) * atom_layout * warp_inner_layout).task_shape) * 4  # 4 types per float32, todo: update when support other data type
            if used_smem_bytes > max_smem_bytes_per_block:
                continue
            pairs.append((a, b))
        self.check(len(pairs) > 0, "Can not find a write-back config")
        pair = max(pairs, key=lambda p: p[0] * p[1])
        self.c_warp_r2s_layout = full_layout(pair) * atom_layout * warp_inner_layout
        c_wb_shape = self.c_warp_r2s_layout.task_shape
        if warp_size <= c_wb_shape[1]:
            self.check(c_wb_shape[1] % warp_size == 0, f"C write back alignment requirement, warp_size = {warp_size}, c_wb_shape = {c_wb_shape}")
            self.c_warp_s2g_layout = full_layout([c_wb_shape[0], c_wb_shape[1] // warp_size]) * row_major([1, warp_size])
        else:
            self.check(warp_size % c_wb_shape[1] == 0 and c_wb_shape[0] % (warp_size // c_wb_shape[1]), f"C write back alignment requirement, warp_size = {warp_size}, c_wb_shape = {c_wb_shape}")
            lines = warp_size // c_wb_shape[1]
            self.c_warp_s2g_layout = full_layout([c_wb_shape[0] // lines, 1]) * row_major([lines, c_wb_shape[1]])

        # derived constants
        used_smem_bytes_per_block = max((block_shape[0] + block_shape[1]) * block_k * 2 * 4,                    # 2 for double buffering, 4 for number of bytes per float32
                                        prod((block_warps_layout * self.c_warp_r2s_layout).task_shape) * 4)     # 4 for number of bytes per float32
        self.check(used_smem_bytes_per_block <= max_smem_bytes_per_block, f"Used shared memory ({used_smem_bytes_per_block} bytes) exceeded the maximum ({max_smem_bytes_per_block} bytes)")
        self.block_size = block_size
        self.block_shape = block_layout.task_shape
        self.block_k = block_k
        self.warp_shape = warp_layout.task_shape
        self.warp_k = warp_k
        self.c_wb_outer = [a // b for a, b in zip(warp_outer, pair)]
        self.c_wb_shape = c_wb_shape
        self.used_num_regs_per_thread = used_num_regs_per_thread
        self.used_smem_bytes_per_block = used_smem_bytes_per_block
        # self.used_smem_bytes_per_block = 2048 * 4
        # we muse use dynamic shared memory when we use more than 48 KiBytes shared memory
        # see Appendix 'Compute Capability' in CUDA C Programming Guide <https://docs.nvidia.com/cuda/pdf/CUDA_C_Programming_Guide.pdf>
        self.use_dynamic_smem = (used_smem_bytes_per_block > 48 * 1024)
        self.min_thread_blocks = resident_blocks
        # self.use_dynamic_smem = False

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
            ('mtb', self.min_thread_blocks)
        ]

    def derived_keys(self) -> List[Tuple[str, Union[int, float, str]]]:
        return [
            ('bx', self.block_shape[0]),
            ('by', self.block_shape[1]),
            ('regs', self.used_num_regs_per_thread),
            ('smem', self.used_smem_bytes_per_block),
        ]

    def __str__(self):
        return 'overall_{}x{}x{}_blcok_warps_{}x{}_outer_{}_{}_middle_{}x{}_inner_{}x{}_warpk_{}_atom_{}_min_blocks_{}'.format(
            *self.block_layout.task_shape, self.block_warps_k * self.warp_k, *self.block_warps, *self.warp_outer, *self.atom_layout.task_shape, *self.warp_inner,
            self.warp_k, self.atom_layout_name, self.min_thread_blocks
        )

    def check(self, cond, msg: str = ""):
        if not cond:
            raise NotSupportedError(msg)

    @staticmethod
    def schedules(space_level: int = 0):
        settings = []
        if space_level == 0:
            settings.append(MatmulSchedule())
        elif space_level == 1:
            for inner_m, inner_n in [[4, 4], [4, 8], [8, 4]]:
                for outer_m, outer_n in [[1, 1], [1, 2], [2, 1], [2, 2]]:
                    for block_warps_k, warp_k in [[4, 1], [8, 1]]:
                        for block_warps_m, block_warps_n in [[2, 2], [2, 4], [4, 2]]:
                            for name, atom_layout in [('row_4x8', TaskLayout.row_major((4, 8))), ('custom_4x8', CustomTaskLayout())]:
                                try:
                                    settings.append(MatmulSchedule(
                                        block_warps_k=block_warps_k,
                                        warp_k=warp_k,
                                        block_warps=[block_warps_m, block_warps_n],
                                        warp_outer=[outer_m, outer_n],
                                        atom_layout=atom_layout,
                                        atom_layout_name=name,
                                        warp_inner=[inner_m, inner_n]
                                    ))
                                except NotSupportedError as e:
                                    pass
        else:
            raise NotImplementedError()
        return settings


@register_impl('cuda_grid_static_matmul_implementer')
class CudaGridStaticMatmulImplementer(Implementer):
    def __init__(self):
        # const definition
        self.task_m = any_const_int()
        self.task_n = any_const_int()
        self.task_k = any_const_int()

        # inputs
        A = TensorInput('A', dtype=scalar_type('float32'), shape=[None, None])
        B = TensorInput('B', dtype=scalar_type('float32'), shape=[None, None])

        # compute
        i, j, k = var('i'), var('j'), var('k')
        computation = TensorCompute(name='C',
                                    shape=[self.task_m, self.task_n],
                                    axes=[i, j],
                                    value=ReduceCompute(
                                        value=A[i, k] * B[k, j],
                                        shape=[self.task_k],
                                        axes=[k],
                                        reduce_type=None)
                                    )

        # inputs and output types
        self.A_type = TensorType(Scope('global'), scalar_type('float32'))
        self.B_type = TensorType(Scope('global'), scalar_type('float32'))
        self.C_type = TensorType(Scope('global'), scalar_type('float32'))

        # pattern
        self.pattern = TaskPattern(
            compute_pattern=computation,
            required_params=[A, B, computation],
            required_param_types=[self.A_type, self.B_type, self.C_type],
            allow_tensor_extra_params=False,
            worker=Grid()
        )

    def priority(self) -> int:
        return 2

    def task_pattern(self) -> TaskPattern:
        return self.pattern

    def implement(self, task: Task, match: Mapping[Node, Node]) -> IRModule:
        schedules = MatmulSchedule.schedules(space_level=0)
        ir_modules = []
        for schedule in schedules:
            ir_modules.append(self.implement_schedule(task, match, schedule))
        return self.resolve(task=task,
                            match=match,
                            schedules=schedules,
                            ir_modules=ir_modules,
                            task_label='matmul_{}x{}x{}'.format(*[int(match[v]) for v in [self.task_m, self.task_n, self.task_k]]),
                            parallel=True,
                            verbose=True)

    def implement_schedule(self, task: Task, match: Mapping[Node, Any], schedule: MatmulSchedule) -> IRModule:
        ir_module = IRModule()
        sch = schedule

        task_m = int(match[self.task_m])
        task_n = int(match[self.task_n])
        task_k = int(match[self.task_k])
        A_dtype = match[self.A_type].scalar_type
        B_dtype = match[self.B_type].scalar_type
        C_dtype = match[self.C_type].scalar_type

        grid_blocks_layout: TaskLayout = row_major_layout(*[(a + b - 1) // b for a, b in zip([task_m, task_n], sch.block_shape)])

        # define function
        with FunctionBuilder(name=task.name + '.grid',
                             worker=Grid(grid_dim=grid_blocks_layout.num_workers,
                                         block_dim=sch.block_size,
                                         dynamic_smem_bytes=sch.used_smem_bytes_per_block if sch.use_dynamic_smem else 0,
                                         min_blocks=sch.min_thread_blocks),
                             label=str(sch)) as fb:
            sb = StmtBuilder()

            # declare params
            gmem_A = Var('A', match[self.A_type])
            gmem_B = Var('B', match[self.B_type])
            gmem_C = Var('C', match[self.C_type])
            fb.extend_params([gmem_A, gmem_B, gmem_C])

            # declare local variables
            smem_A = Var('smem_A', TensorPointerType('shared', A_dtype, layout=StridesLayout.from_shape([2, sch.block_shape[0], sch.block_k], perm=[0, 2, 1])))
            smem_B = Var('smem_B', TensorPointerType('shared', B_dtype, layout=StridesLayout.from_shape([2, sch.block_k, sch.block_shape[1]], perm=[0, 1, 2])))
            smem_C = Var('smem_C', TensorPointerType('shared', C_dtype, layout=StridesLayout.row_major((sch.block_warps_layout * sch.c_warp_s2g_layout).task_shape)))
            if sch.use_dynamic_smem:
                # 'extern __shared__ uint8_t smem_storage[];' in c code
                smem_storage = Var('smem_storage', PointerType(base_type=scalar_type('uint8'), specifiers=['extern', '__shared__'], use_bracket=True))
            else:
                smem_storage = Var('smem_storage', TensorType('shared', dtype='uint8', shape=[sch.used_smem_bytes_per_block], layout=[1]))
            smem_A_bytes = simplify_to_int(smem_A.type.tensor_type.storage_bytes())
            fb.extend_local_vars([smem_A, smem_B, smem_C, smem_storage])
            sb += AssignStmt(smem_A, Cast(~smem_storage[0], PointerType(A_dtype)))
            sb += AssignStmt(smem_B, Cast(~(smem_storage[smem_A_bytes]), PointerType(B_dtype)))
            sb += AssignStmt(smem_C, Cast(~(smem_storage[0]), PointerType(C_dtype)))

            # declare A, B, C registers
            regs_A = Var('regs_A', TensorType('register', A_dtype, layout=StridesLayout.row_major([2]) + schedule.regs_a_layout))
            regs_B = Var('regs_B', TensorType('register', B_dtype, layout=StridesLayout.row_major([2]) + schedule.regs_b_layout))
            regs_C = Var('regs_C', TensorType('register', C_dtype, layout=schedule.regs_c_layout))
            regs_A_ldg = Var('regs_A_ldg', TensorType(scope='register', dtype=A_dtype, layout=schedule.regs_a_ldg_layout))
            regs_B_ldg = Var('regs_B_ldg', TensorType(scope='register', dtype=B_dtype, layout=schedule.regs_b_ldg_layout))
            fb.extend_local_vars([regs_A, regs_B, regs_C, regs_A_ldg, regs_B_ldg])

            with sb.lets(['bi', 'bj'], grid_blocks_layout(block_idx())[0]) as (bi, bj):
                block_k_tiles = (task_k + sch.block_k - 1) // sch.block_k
                first_k_tile = task_k - (block_k_tiles - 1) * sch.block_k
                block_offset = [idx * dim for idx, dim in zip([bi, bj], sch.block_shape)]
                # transfer first tile
                sb += self.copy(gmem_A[block_offset[0]:, :], regs_A_ldg, schedule.a_g2s_layout, src_predicate=lambda i, k: And.join(block_offset[0] + i < task_m, k < first_k_tile))
                sb += self.copy(regs_A_ldg, smem_A[0], layout=schedule.a_g2s_layout)
                sb += self.copy(gmem_B[:, block_offset[1]:], regs_B_ldg, schedule.b_g2s_layout, src_predicate=lambda k, j: And.join(k < first_k_tile, block_offset[1] + j < task_n))
                sb += self.copy(regs_B_ldg, smem_B[0], layout=schedule.b_g2s_layout)
                sb += syncthreads()
                sb += self.copy(smem_A[0], regs_A[0], schedule.a_s2r_layout)
                sb += self.copy(smem_B[0], regs_B[0], schedule.b_s2r_layout)
                sb += syncthreads()
                # init regs c
                sb += self.init(regs_C, 0.0, schedule.block_layout)
                with sb.for_loop('k0', block_k_tiles - 1) as k0:
                    block_offset_k = k0 * sch.block_k + first_k_tile
                    with sb.for_loop('k1', sch.block_warps_k) as k1:
                        with sb.if_then(Equal(k1, sch.block_warps_k - 1)):
                            sb += self.copy(regs_A_ldg, smem_A[(k0 + 1) % 2], schedule.a_g2s_layout)
                            sb += self.copy(regs_B_ldg, smem_B[(k0 + 1) % 2], schedule.b_g2s_layout)
                            sb += syncthreads()
                            sb += self.copy(smem_A[(k0 + 1) % 2], regs_A[(k1 + 1) % 2], schedule.a_s2r_layout)
                            sb += self.copy(smem_B[(k0 + 1) % 2], regs_B[(k1 + 1) % 2], schedule.b_s2r_layout)
                        with sb.otherwise():
                            sb += self.copy(smem_A[k0 % 2, :, k1 + 1:], regs_A[(k1 + 1) % 2], schedule.a_s2r_layout)
                            sb += self.copy(smem_B[k0 % 2, k1 + 1:, :], regs_B[(k1 + 1) % 2], schedule.b_s2r_layout)
                        with sb.if_then(Equal(k1, 0)):
                            sb += self.copy(gmem_A[block_offset[0]:, block_offset_k:], regs_A_ldg, schedule.a_g2s_layout, src_predicate=lambda i, _: block_offset[0] + i < task_m)
                            sb += self.copy(gmem_B[block_offset_k:, block_offset[1]:], regs_B_ldg, schedule.b_g2s_layout, src_predicate=lambda _, j: block_offset[1] + j < task_n)
                        sb += self.mma(regs_A[k1 % 2], regs_B[k1 % 2], regs_C, schedule)
                with sb.let('block_k_tile', block_k_tiles - 1) as k0:
                    with sb.for_loop('warp_k_tile', sch.block_warps_k) as k1:
                        with sb.if_then(k1 < sch.block_warps_k - 1):
                            sb += self.copy(smem_A[k0 % 2, :, k1 + 1:], regs_A[(k1 + 1) % 2], schedule.a_s2r_layout)
                            sb += self.copy(smem_B[k0 % 2, k1 + 1:, :], regs_B[(k1 + 1) % 2], schedule.b_s2r_layout)
                        sb += self.mma(regs_A[k1 % 2], regs_B[k1 % 2], regs_C, schedule)
                with sb.for_loop('i', sch.c_wb_outer[0]) as i:
                    with sb.for_loop('j', sch.c_wb_outer[1]) as j:
                        warp_indices = sch.block_warps_layout(thread_idx() // 32)[0]
                        regs_warp_offset = [wid * wdim + pid * pdim for wid, wdim, pid, pdim in zip(warp_indices, sch.warp_layout.task_shape, [i, j], sch.c_wb_shape)]
                        smem_warp_offset = [idx * dim for idx, dim in zip(warp_indices, sch.c_wb_shape)]
                        gmem_warp_offset = [bo + ro for bo, ro in zip(block_offset, regs_warp_offset)]
                        sb += syncthreads()
                        sb += self.copy(src=regs_C[regs_warp_offset[0]:, regs_warp_offset[1]:], dst=smem_C[smem_warp_offset[0]:, smem_warp_offset[1]:], layout=schedule.c_warp_r2s_layout)
                        sb += syncthreads()
                        sb += self.copy(src=smem_C[smem_warp_offset[0]:, smem_warp_offset[1]:], dst=gmem_C[gmem_warp_offset[0]:, gmem_warp_offset[1]:], layout=schedule.c_warp_s2g_layout,
                                        dst_predicate=lambda ii, jj: And.join(gmem_warp_offset[0] + ii < task_m, gmem_warp_offset[1] + jj < task_n))
            # set body
            fb.set_body(sb.finish())

        func = fb.get()
        ir_module.add(func.name, func)
        return ir_module

    def init(self, dst, init_value, layout):
        sb = StmtBuilder()
        for indices in layout(thread_idx()):
            sb += BufferStoreStmt(dst, indices, init_value)
        return sb.finish()

    def copy(self, src, dst, layout, src_predicate=None, dst_predicate=None):
        sb = StmtBuilder()
        for indices in layout(thread_idx()):
            value = src.__getitem__(indices)
            if src_predicate:
                value = if_then_else(src_predicate(*indices), value, 0.0)
            stmt = BufferStoreStmt(dst, indices, value)
            if dst_predicate:
                stmt = IfStmt(dst_predicate(*indices), stmt)
            sb += stmt
        return sb.finish()

    def mma(self, a, b, c, schedule):
        layout = schedule.block_layout
        sb = StmtBuilder()
        for i, j in layout(thread_idx()):
            for k in range(schedule.warp_k):
                sb += BufferStoreStmt(c, [i, j], c[i, j] + a[i, k] * b[k, j])
        return sb.finish()


if __name__ == '__main__':
    schedules = MatmulSchedule.schedules(space_level=1)
    print(len(schedules))
