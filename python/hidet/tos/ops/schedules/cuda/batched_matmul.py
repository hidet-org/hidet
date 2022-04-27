import itertools
from typing import Mapping, List, Any, Tuple, Union

from hidet.ir.builders import FunctionBuilder, StmtBuilder
from hidet.ir.dialects.lowlevel import TensorPointerType, PointerType
from hidet.ir.expr import var, Var, And, Equal, Cast, if_then_else
from hidet.ir.func import IRModule
from hidet.ir.functors import simplify_to_int
from hidet.ir.layout import TaskLayout, row_major_layout, DataLayout, StridesLayout
from hidet.ir.node import Node
from hidet.ir.primitives import syncthreads, thread_idx, block_idx
from hidet.ir.stmt import AssignStmt, BufferStoreStmt, IfStmt
from hidet.ir.type import scalar_type, TensorType, Scope, tensor_type
from hidet.utils import Timer, cuda, factor, prod
from hidet.tos.ops.definitions.matmul import MatmulTask
from hidet.tos.ops.schedules.cuda.common import resolve_ir_modules
from hidet.tos.ops.schedules.common import params_from_task, inputs_from_task, outputs_from_task, Schedule, NotSupportedError


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
        used_smem_bytes_per_block = max((block_shape[0] + block_shape[1]) * block_k * 2 * 4,  # 2 for double buffering, 4 for number of bytes per float32
                                        prod((block_warps_layout * self.c_warp_r2s_layout).task_shape) * 4)  # 4 for number of bytes per float32
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
        elif space_level == 1 or space_level == 2:
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


def batched_matmul_cuda_schedule(task: MatmulTask, space_level: int = 0) -> IRModule:
    schedules = MatmulSchedule.schedules(space_level=space_level)
    ir_modules = []
    for schedule in schedules:
        ir_modules.append(batched_matmul_cuda_with_given_schedule(task, schedule))
    return resolve_ir_modules(
        ir_modules=ir_modules,
        schedules=schedules,
        task=task,
        task_label='batched_matmul_{}x{}x{}x{}'.format(task.batch_size, task.m_size, task.k_size, task.n_size),
        parallel=True,
        verbose=True
    )


def batched_matmul_cuda_with_given_schedule(task: MatmulTask, schedule: MatmulSchedule) -> IRModule:
    if len(task.epilogues) > 0:
        raise NotImplementedError()
    ir_module = IRModule(task=task)
    sch = schedule

    a_dtype = task.inputs[0].data_type.scalar_type
    b_dtype = task.inputs[1].data_type.scalar_type
    c_dtype = task.outputs[0].data_type.scalar_type

    batch_size = task.batch_size
    m_size, k_size, n_size = task.m_size, task.k_size, task.n_size

    m_tile_size, n_tile_size = sch.block_shape
    m_tiles = (m_size + m_tile_size - 1) // m_tile_size
    n_tiles = (n_size + n_tile_size - 1) // n_tile_size
    grid_blocks_layout: TaskLayout = TaskLayout.row_major([m_tiles, n_tiles])

    # define function
    with FunctionBuilder(
            name=task.name + '.grid',
            kind='cuda_kernel',
            grid_dim=(grid_blocks_layout.num_workers, batch_size),
            block_dim=sch.block_size,
            dynamic_smem_bytes=sch.used_smem_bytes_per_block if sch.use_dynamic_smem else 0,
            min_blocks=sch.min_thread_blocks,
            label=str(sch)) as fb:
        sb = StmtBuilder()

        # declare params
        params = params_from_task(task)
        inputs = inputs_from_task(task, params)
        outputs = outputs_from_task(task, params)
        gmem_a, gmem_b = inputs
        gmem_c, = outputs
        fb.extend_params(params)

        # declare local variables
        smem_a = Var('smem_a', TensorPointerType('shared', a_dtype, layout=StridesLayout.from_shape([2, sch.block_shape[0], sch.block_k], perm=[0, 2, 1])))
        smem_b = Var('smem_b', TensorPointerType('shared', b_dtype, layout=StridesLayout.from_shape([2, sch.block_k, sch.block_shape[1]], perm=[0, 1, 2])))
        smem_c = Var('smem_c', TensorPointerType('shared', c_dtype, layout=StridesLayout.row_major((sch.block_warps_layout * sch.c_warp_s2g_layout).task_shape)))
        if sch.use_dynamic_smem:
            # 'extern __shared__ uint8_t smem_storage[];' in c code
            smem_storage = Var('smem_storage', PointerType(base_type=scalar_type('uint8'), specifiers=['extern', '__shared__'], use_bracket=True))
        else:
            smem_storage = Var('smem_storage', tensor_type('shared', dtype='uint8', shape=[sch.used_smem_bytes_per_block]))
        smem_A_bytes = simplify_to_int(smem_a.type.tensor_type.storage_bytes())
        fb.extend_local_vars([smem_a, smem_b, smem_c, smem_storage])
        sb += AssignStmt(smem_a, Cast(~smem_storage[0], PointerType(a_dtype)))
        sb += AssignStmt(smem_b, Cast(~(smem_storage[smem_A_bytes]), PointerType(b_dtype)))
        sb += AssignStmt(smem_c, Cast(~(smem_storage[0]), PointerType(c_dtype)))

        # declare a, b, c registers
        regs_a = Var('regs_A', tensor_type('register', a_dtype, layout=StridesLayout.row_major([2]) + schedule.regs_a_layout))
        regs_b = Var('regs_B', tensor_type('register', b_dtype, layout=StridesLayout.row_major([2]) + schedule.regs_b_layout))
        regs_c = Var('regs_C', tensor_type('register', c_dtype, layout=schedule.regs_c_layout))
        regs_a_ldg = Var('regs_A_ldg', tensor_type(scope='register', dtype=a_dtype, layout=schedule.regs_a_ldg_layout))
        regs_b_ldg = Var('regs_B_ldg', tensor_type(scope='register', dtype=b_dtype, layout=schedule.regs_b_ldg_layout))
        fb.extend_local_vars([regs_a, regs_b, regs_c, regs_a_ldg, regs_b_ldg])

        with sb.lets(['bi', 'bj'], grid_blocks_layout(block_idx())[0]) as (bi, bj):
            block_k_tiles = (k_size + sch.block_k - 1) // sch.block_k
            first_k_tile = k_size - (block_k_tiles - 1) * sch.block_k
            block_offset = [idx * dim for idx, dim in zip([bi, bj], sch.block_shape)]
            # transfer first tile
            sb += copy(gmem_a[block_idx('y'), block_offset[0]:, :], regs_a_ldg, schedule.a_g2s_layout, src_predicate=lambda i, k: And.join(block_offset[0] + i < m_size, k < first_k_tile))
            sb += copy(regs_a_ldg, smem_a[0], layout=schedule.a_g2s_layout)
            sb += copy(gmem_b[block_idx('y'), :, block_offset[1]:], regs_b_ldg, schedule.b_g2s_layout, src_predicate=lambda k, j: And.join(k < first_k_tile, block_offset[1] + j < n_size))
            sb += copy(regs_b_ldg, smem_b[0], layout=schedule.b_g2s_layout)
            sb += syncthreads()
            sb += copy(smem_a[0], regs_a[0], schedule.a_s2r_layout)
            sb += copy(smem_b[0], regs_b[0], schedule.b_s2r_layout)
            sb += syncthreads()
            # init regs c
            sb += init(regs_c, 0.0, schedule.block_layout)
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
                        sb += copy(smem_a[k0 % 2, :, k1 + 1:], regs_a[(k1 + 1) % 2], schedule.a_s2r_layout)
                        sb += copy(smem_b[k0 % 2, k1 + 1:, :], regs_b[(k1 + 1) % 2], schedule.b_s2r_layout)
                    with sb.if_then(Equal(k1, 0)):
                        sb += copy(gmem_a[block_idx('y'), block_offset[0]:, block_offset_k:], regs_a_ldg, schedule.a_g2s_layout, src_predicate=lambda i, _: block_offset[0] + i < m_size)
                        sb += copy(gmem_b[block_idx('y'), block_offset_k:, block_offset[1]:], regs_b_ldg, schedule.b_g2s_layout, src_predicate=lambda _, j: block_offset[1] + j < n_size)
                    sb += mma(regs_a[k1 % 2], regs_b[k1 % 2], regs_c, schedule)
            with sb.let('block_k_tile', block_k_tiles - 1) as k0:
                with sb.for_loop('warp_k_tile', sch.block_warps_k) as k1:
                    with sb.if_then(k1 < sch.block_warps_k - 1):
                        sb += copy(smem_a[k0 % 2, :, k1 + 1:], regs_a[(k1 + 1) % 2], schedule.a_s2r_layout)
                        sb += copy(smem_b[k0 % 2, k1 + 1:, :], regs_b[(k1 + 1) % 2], schedule.b_s2r_layout)
                    sb += mma(regs_a[k1 % 2], regs_b[k1 % 2], regs_c, schedule)
            with sb.for_loop('i', sch.c_wb_outer[0]) as i:
                with sb.for_loop('j', sch.c_wb_outer[1]) as j:
                    warp_indices = sch.block_warps_layout(thread_idx() // 32)[0]
                    regs_warp_offset = [wid * wdim + pid * pdim for wid, wdim, pid, pdim in zip(warp_indices, sch.warp_layout.task_shape, [i, j], sch.c_wb_shape)]
                    smem_warp_offset = [idx * dim for idx, dim in zip(warp_indices, sch.c_wb_shape)]
                    gmem_warp_offset = [bo + ro for bo, ro in zip(block_offset, regs_warp_offset)]
                    sb += syncthreads()
                    sb += copy(src=regs_c[regs_warp_offset[0]:, regs_warp_offset[1]:], dst=smem_c[smem_warp_offset[0]:, smem_warp_offset[1]:], layout=schedule.c_warp_r2s_layout)
                    sb += syncthreads()
                    sb += copy(src=smem_c[smem_warp_offset[0]:, smem_warp_offset[1]:], dst=gmem_c[block_idx('y'), gmem_warp_offset[0]:, gmem_warp_offset[1]:], layout=schedule.c_warp_s2g_layout,
                               dst_predicate=lambda ii, jj: And.join(gmem_warp_offset[0] + ii < m_size, gmem_warp_offset[1] + jj < n_size))
        # set body
        fb.set_body(sb.finish())

    func = fb.get()
    ir_module.add(func.name, func)
    return ir_module


def init(dst, init_value, layout):
    sb = StmtBuilder()
    for indices in layout(thread_idx()):
        sb += BufferStoreStmt(dst, indices, init_value)
    return sb.finish()


def copy(src, dst, layout, src_predicate=None, dst_predicate=None):
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


def mma(a, b, c, schedule):
    layout = schedule.block_layout
    sb = StmtBuilder()
    for i, j in layout(thread_idx()):
        for k in range(schedule.warp_k):
            sb += BufferStoreStmt(c, [i, j], c[i, j] + a[i, k] * b[k, j])
    return sb.finish()


if __name__ == '__main__':
    schedules = MatmulSchedule.schedules(space_level=1)
    print(len(schedules))
