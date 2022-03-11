from typing import Mapping, List, Union, Any, Callable, Sequence
import numpy as np
import contextlib

from hidet.implement.implementer import Implementer, register_impl, NotSupportedError
from hidet.implement.search_space import ProductSpace, AtomSpace, SpaceChoice
from hidet.ir.builders import TaskBuilder, FunctionBuilder, StmtBuilder
from hidet.ir.dialects.compute import TensorInput, TensorCompute, compute, ReduceCompute, reduce_sum, ScalarInput
from hidet.ir.dialects.lowlevel import Address, TensorPointerType, Cast, PointerType
from hidet.ir.dialects.pattern import TaskPattern, any_const_int
from hidet.ir.functors import simplify, simplify_to_int
from hidet.ir.expr import Expr, Call, TensorElement, var, tensor_var, convert, Var, And, IfThenElse, Or, is_tensor, get_tensor_layout, PyScalar, Condition, Equal, Constant
from hidet.ir.func import IRModule
from hidet.ir.layout import TaskLayout, row_major_layout, full_layout, DataLayout, StridesLayout
from hidet.ir.node import Node
from hidet.ir.primitives import syncthreads, thread_idx, block_idx
from hidet.ir.stmt import ForStmt, AssignStmt, ReturnStmt, BufferStoreStmt, Stmt
from hidet.ir.task import Task, ThreadBlock, Warp, Grid
from hidet.ir.type import scalar_type, TensorType, Scope
from hidet.implement.common import transfer_task, init_task, transfer_predicated_task, predicated_transfer_task, bounded_transfer_task, transfer_bounded_task
from hidet.backend import batch_build, BuildInstance, build
from hidet.utils import Timer

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


class Schedule:
    def __init__(self,
                 block_warps_k=8,
                 warp_k=1,
                 block_warps=(4, 2),
                 outer=(2, 2),
                 atom_layout=CustomTaskLayout(),
                 atom_layout_name='custom_4x8',
                 inner=(4, 4),
                 min_thread_blocks=2):
        self.block_warps_k = block_warps_k
        self.warp_k = warp_k
        self.block_warps = block_warps
        self.outer = outer
        self.atom_layout = atom_layout
        self.inner = inner
        self.atom_layout_name = atom_layout_name
        self.min_thread_blocks = min_thread_blocks
        block_k = block_warps_k * warp_k
        # dependent variables
        warp_m = outer[0] * atom_layout.task_shape[0] * inner[0]
        warp_n = outer[1] * atom_layout.task_shape[1] * inner[1]
        block_m = block_warps[0] * warp_m
        block_n = block_warps[1] * warp_n
        atom_shape = atom_layout.task_shape

        warp_size = 32
        num_warps = block_warps[0] * block_warps[1]
        block_size = num_warps * warp_size
        self.block_size = block_size
        self.check(block_warps_k % warp_k == 0 and (block_warps_k // warp_k) % 2 == 0)  # double buffering index requirement
        # (warp_size % block_warps_k != 0 and block_warps_k % warp_size == 0)  # this is another case we might handle
        self.check(warp_size % block_warps_k == 0)  # gmem -> regs requirement. block_k candidates: 2, 4, 8, 16, 32
        self.check(block_m % (block_size // block_warps_k) == 0)  # A: gmem -> regs alignment requirement
        self.check(block_n % (block_size // block_warps_k) == 0)  # B: gmem -> regs alignment requirement
        self.check(warp_k == 1)

        # task layouts
        self.block_layout = row_major_layout(*block_warps)
        self.warp_layout = full_layout(*outer) * atom_layout * full_layout(*inner)
        self.a_g2r_r2s_layout = row_major_layout(block_size // block_warps_k, block_warps_k) * full_layout(block_m // (block_size // block_warps_k), 1)
        self.b_g2r_r2s_layout = full_layout(1, block_n // (block_size // block_warps_k)) * row_major_layout(block_warps_k, block_size // block_warps_k)
        self.a_s2r_layout: TaskLayout = (self.block_layout * full_layout(outer[0], 1) * atom_layout * full_layout(inner[0], 1)).projection({1: 0})
        self.b_s2r_layout: TaskLayout = (self.block_layout * full_layout(1, outer[1]) * atom_layout * full_layout(1, inner[1])).projection({0: 0})
        self.ab2c_layout = self.block_layout * self.warp_layout
        self.c_r2s_outer_inner = self.get_c_write_back_inner()
        self.warp_c_wb_layout = full_layout(*self.c_r2s_outer_inner) * atom_layout * full_layout(*inner)
        self.check(self.warp_c_wb_layout.task_shape[1] % 32 == 0)
        self.c_r2s_layout = self.block_layout * self.warp_c_wb_layout
        self.c_s2g_layout = self.block_layout * full_layout(self.warp_c_wb_layout.task_shape[0], self.warp_c_wb_layout.task_shape[1] // warp_size) * row_major_layout(1, warp_size)
        self.c_r2s_outer_outer = [outer[0] // self.c_r2s_outer_inner[0], outer[1] // self.c_r2s_outer_inner[1]]

        self.wb_warp_shape = [atom_shape[0] * inner[0] * self.c_r2s_outer_inner[0], atom_shape[1] * inner[1] * self.c_r2s_outer_inner[1]]

        # regs data layout
        self.regs_a_layout = DataLayout.local((block_warps[0], 1)) * StridesLayout.row_major((outer[0], 1)) * DataLayout.local((atom_shape[0], 1)) * StridesLayout.row_major((inner[0], 1))
        self.regs_b_layout = DataLayout.local((1, block_warps[1])) * StridesLayout.row_major((1, outer[1])) * DataLayout.local((1, atom_shape[1])) * StridesLayout.row_major((1, inner[1]))
        self.regs_c_layout = DataLayout.local(block_warps) * StridesLayout.row_major(outer) * DataLayout.local(atom_shape) * StridesLayout.row_major(inner)
        self.regs_a_ldg_layout = DataLayout.local((block_size // block_k, block_k)) * DataLayout.row_major((block_m // (block_size // block_k), 1))
        self.regs_b_ldg_layout = DataLayout.row_major((1, block_n // (block_size // block_k))) * DataLayout.local((block_k, block_size // block_k))

    def __str__(self):
        return 'overall_{}x{}x{}_blcok_warps_{}x{}_outer_{}_{}_middle_{}x{}_inner_{}x{}_warpk_{}_atom_{}_min_blocks_{}'.format(
            *self.ab2c_layout.task_shape, self.block_warps_k * self.warp_k, *self.block_warps, *self.outer, *self.atom_layout.task_shape, *self.inner,
            self.warp_k, self.atom_layout_name, self.min_thread_blocks
        )

    def get_c_write_back_inner(self):
        block_size = self.block_warps[0] * self.block_warps[1] * 32
        block_m = self.block_warps[0] * self.warp_layout.task_shape[0]
        block_n = self.block_warps[1] * self.warp_layout.task_shape[1]
        found_regs_size = 0
        c_r2s_outer_inner = None
        for a in range(1, self.outer[0] + 1):
            if self.outer[0] % a != 0:
                continue
            for b in range(1, self.outer[1] + 1):
                if self.outer[1] % b != 0:
                    continue
                regs_size = a * b * self.inner[0] * self.inner[1] * block_size
                smem_size = (block_m + block_n) * self.block_warps_k * 2
                if regs_size > smem_size:
                    continue
                if found_regs_size == 0 or found_regs_size <= regs_size:
                    found_regs_size = regs_size
                    c_r2s_outer_inner = [a, b]
        if found_regs_size == 0:
            raise NotSupportedError('can not use smem to help write c.')
        return c_r2s_outer_inner

    def check(self, cond):
        if not cond:
            raise NotSupportedError()


matmul_settings = [Schedule()]
atom_layouts = [
    ('row_4x8', TaskLayout.row_major((4, 8))),
    ('col_4x8', TaskLayout.row_major((8, 4))),
    # ('custom_4x8', TaskLayout(num_workers=32, task_shape=(4, 8), worker2task=lambda w: [(w // 16 * 2 + w % 2, w // 2 % 8)])),
    ('custom_4x8', CustomTaskLayout()),
]


def setup_matmul_settings(use_default=False):
    if use_default:
        return [Schedule()]
    else:
        settings = []
        for inner_m, inner_n in [[4, 4], [8, 8], [4, 8], [8, 4]]:
            for outer_m, outer_n in [[1, 1], [1, 2], [2, 1], [2, 2]]:
                for block_warps_k, warp_k in [[2, 1], [4, 1], [8, 1], [12, 1]]:
                    for block_warps_m, block_warps_n in [[2, 2], [2, 4], [4, 2], [4, 2]]:
                        for min_thread_blocks in [1, 2]:
                            for name, atom_layout in atom_layouts:
                                try:
                                    settings.append(Schedule(
                                        block_warps_k=block_warps_k,
                                        warp_k=warp_k,
                                        block_warps=[block_warps_m, block_warps_n],
                                        outer=[outer_m, outer_n],
                                        atom_layout=atom_layout,
                                        atom_layout_name=name,
                                        inner=[inner_m, inner_n],
                                        min_thread_blocks=min_thread_blocks
                                    ))
                                except NotSupportedError:
                                    pass
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
        # search space
        settings = setup_matmul_settings(use_default=True)
        ir_modules = []
        for setting in settings:
            ir_modules.append(self.implement_schedule(task, match, setting))
        if len(ir_modules) > 1:
            return self.resolve(task, match, ir_modules)
        else:
            assert len(ir_modules) == 1
            return ir_modules[0]

    def resolve(self, task, match, ir_modules: List[IRModule]) -> IRModule:
        from hidet.runtime.value import dummy_inputs_from_task
        parallel = True
        task_label = 'matmul_{}x{}x{}'.format(int(match[self.task_m]), int(match[self.task_n]), int(match[self.task_k]))
        with Timer('Resolving', verbose=False):
            build_instances = [BuildInstance(ir_module=ir_module, output_dir=f'./outs/resolve/{task_label}/{idx}', keep_ir=False, nvcc_keep=True, verbose=False) for idx, ir_module in enumerate(ir_modules)]
            compiled_modules = batch_build(build_instances, parallel=parallel, verbose=False)
            dummy_inputs = dummy_inputs_from_task(task)
            best_latency = None
            best_ir_module = None
            latencies = []
            for ir_module, compiled_module in zip(ir_modules, compiled_modules):
                repeat_latency = compiled_module[task.name].profile(*dummy_inputs, warmup=2, number=1, repeat=10)
                # print(repeat_latency)
                latency = np.median(repeat_latency)
                latencies.append(latency)
                if best_latency is None or best_latency > latency:
                    best_latency = latency
                    best_ir_module = ir_module
            # pairs = [(latency, label) for latency, label in zip(latencies, [ir_module.functions[task.name + '_grid'].attrs['label'] for ir_module in ir_modules])]
            # for latency, label in sorted(pairs, key=lambda p: p[0]):
            #     print('{:>120}: {:.3f}'.format(label, latency))
            return best_ir_module

    def implement_schedule(self, task: Task, match: Mapping[Node, Any], schedule: Schedule) -> IRModule:
        ir_module = IRModule()

        # task-related constants
        task_m = int(match[self.task_m])
        task_n = int(match[self.task_n])
        task_k = int(match[self.task_k])

        # space-related constants
        block_size = schedule.block_size
        block_layout = schedule.ab2c_layout
        block_warps_layout: TaskLayout = schedule.block_layout
        warp_layout: TaskLayout = schedule.warp_layout
        block_shape = schedule.ab2c_layout.task_shape
        grid_blocks_layout: TaskLayout = row_major_layout(*[(a + b - 1) // b for a, b in zip([task_m, task_n], block_shape)])

        # block_k = setting.block_k

        block_m, block_n = schedule.ab2c_layout.task_shape
        warp_m, warp_n = warp_layout.task_shape
        warp_k = schedule.warp_k
        block_k = schedule.block_warps_k * warp_k

        wb_warp_shape = schedule.wb_warp_shape

        # task-related variables
        gmem_a_type: TensorType = match[self.A_type]
        gmem_b_type: TensorType = match[self.B_type]
        gmem_c_type: TensorType = match[self.C_type]
        gmem_a_type = gmem_a_type.split(dim2factor={0: block_shape[0]})
        gmem_b_type = gmem_b_type.split(dim2factor={1: block_shape[1]}).reorder([1, 0, 2])
        gmem_c_type = gmem_c_type.split(dim2factor={0: block_shape[0], 1: block_shape[1]}).reorder([0, 2, 1, 3])

        A_dtype = gmem_a_type.scalar_type
        B_dtype = gmem_b_type.scalar_type
        C_dtype = gmem_c_type.scalar_type

        # declare inputs and outputs and their types shared by all subtasks
        smem_A_type = TensorType(scope='shared', dtype=A_dtype, shape=[block_m, block_k], layout=[1, block_m])  # column major, TODO: add to search space
        smem_B_type = TensorType(scope='shared', dtype=B_dtype, shape=[block_k, block_n], layout=[block_n, 1])  # row major
        regs_A_type = TensorType(scope='register', dtype=A_dtype, layout=schedule.regs_a_layout)
        regs_B_type = TensorType(scope='register', dtype=B_dtype, layout=schedule.regs_b_layout)
        regs_C_type = TensorType(scope='register', dtype=C_dtype, layout=schedule.regs_c_layout)
        regs_A_ldg_type = TensorType(scope='register', dtype=A_dtype, layout=schedule.regs_a_ldg_layout)
        regs_B_ldg_type = TensorType(scope='register', dtype=B_dtype, layout=schedule.regs_b_ldg_layout)
        #                                                                  (GM, GN, M, N) -> (M, N)    ->     (BM, WM, BN, WN) ->           (BM, OM, IM, BN, ON, IN) ->                       (OM, ON, BM * IM, BN * IN)
        gmem_C_wb_type = TensorType(scope='global', dtype=C_dtype, layout=gmem_c_type.layout.split({2: warp_m, 3: warp_n}).split({3: wb_warp_shape[0], 5: wb_warp_shape[1]}).fuse([0, 1, 3, 6, [2, 4], [5, 7]]))
        smem_C_wb_type = TensorType(scope='shared', dtype=C_dtype, layout=StridesLayout.row_major(schedule.block_warps) * StridesLayout.row_major(schedule.warp_c_wb_layout.task_shape))
        regs_C_wb_type = TensorType(scope='register', dtype=C_dtype, layout=regs_C_type.layout.split({0: warp_m, 1: warp_n}).split({1: wb_warp_shape[0], 3: wb_warp_shape[1]}).fuse([1, 4, [0, 2], [3, 5]]))

        # define subtasks
        block_k_tiles = (task_k + block_k - 1) // block_k
        first_k_tile = task_k - (block_k_tiles - 1) * block_k

        # define function
        with FunctionBuilder(task.name + '.grid', attrs={'worker': Grid(grid_dim=grid_blocks_layout.num_workers, block_dim=block_size, min_blocks=schedule.min_thread_blocks),
                                                         'label': str(schedule)}) as fb:
            sb = StmtBuilder()

            # declare params
            gmem_A = Var('A', gmem_a_type)
            gmem_B = Var('B', gmem_b_type)
            gmem_C = Var('C', gmem_c_type)
            fb.extend_params([gmem_A, gmem_B, gmem_C])

            # declare local variables
            smem_A_tp = TensorPointerType('shared', A_dtype, layout=StridesLayout.from_shape([2, block_m, block_k], perm=[0, 2, 1]))
            smem_B_tp = TensorPointerType('shared', B_dtype, layout=StridesLayout.from_shape([2, block_k, block_m], perm=[0, 1, 2]))
            smem_C_tp = TensorPointerType('shared', C_dtype, layout=smem_C_wb_type.layout)
            smem_A = Var('smem_A', smem_A_tp)
            smem_B = Var('smem_B', smem_B_tp)
            smem_C = Var('smem_C', smem_C_tp)
            smem_A_bytes = simplify_to_int(smem_A_tp.tensor_type.storage_bytes())
            smem_B_bytes = simplify_to_int(smem_B_tp.tensor_type.storage_bytes())
            smem_C_bytes = simplify_to_int(smem_C_tp.tensor_type.storage_bytes())
            smem_storage = Var('smem_storage', TensorType('shared', dtype='uint8', shape=[max(smem_A_bytes + smem_B_bytes, smem_C_bytes)], layout=[1]))
            fb.extend_local_vars([smem_A, smem_B, smem_C, smem_storage])
            gmem_C_wb = Var('gmem_C_wb', TensorPointerType('global', C_dtype, layout=gmem_C_wb_type.layout))
            fb.extend_local_vars([gmem_C_wb])
            sb += AssignStmt(gmem_C_wb, gmem_C)
            sb += AssignStmt(smem_A, Cast(~smem_storage[0], PointerType(A_dtype)))
            sb += AssignStmt(smem_B, Cast(~(smem_storage[smem_A_bytes]), PointerType(B_dtype)))
            sb += AssignStmt(smem_C, Cast(~(smem_storage[0]), PointerType(C_dtype)))

            # declare A, B, C registers
            regs_A = Var('regs_A', TensorType('register', A_dtype, (2,) + regs_A_type.shape, StridesLayout.row_major([2]) + regs_A_type.layout))
            regs_B = Var('regs_B', TensorType('register', B_dtype, (2,) + regs_B_type.shape, StridesLayout.row_major([2]) + regs_B_type.layout))
            regs_C = Var('regs_C', regs_C_type)
            regs_A_ldg = Var('regs_A_ldg', regs_A_ldg_type)
            regs_B_ldg = Var('regs_B_ldg', regs_B_ldg_type)
            regs_C_wb = Var('regs_C_wb', TensorPointerType('register', C_dtype, layout=regs_C_wb_type.layout))
            sb += AssignStmt(regs_C_wb, regs_C)
            fb.extend_local_vars([regs_A, regs_B, regs_C, regs_A_ldg, regs_B_ldg, regs_C_wb])

            with sb.lets(['block_m', 'block_n'], grid_blocks_layout(block_idx())[0]) as (block_m, block_n):
                # transfer first tile
                sb += self.transfer(dst=regs_A_ldg, src=gmem_A, layout=schedule.a_g2r_r2s_layout, f_src_index=lambda i, k: [block_m, i, k], protect_src=True, src_predicate=lambda i, k: k < first_k_tile)
                sb += self.copy(regs_A_ldg, smem_A[0], layout=schedule.a_g2r_r2s_layout)
                sb += self.transfer(dst=regs_B_ldg, src=gmem_B, layout=schedule.b_g2r_r2s_layout, f_src_index=lambda k, j: [block_n, k, j], protect_src=True, src_predicate=lambda k, j: k < first_k_tile)
                sb += self.copy(regs_B_ldg, smem_B[0], layout=schedule.b_g2r_r2s_layout)
                sb += syncthreads()
                sb += self.copy(smem_A[0], regs_A[0], schedule.a_s2r_layout)
                sb += self.copy(smem_B[0], regs_B[0], schedule.b_s2r_layout)
                sb += syncthreads()
                # init regs c
                sb += self.init(regs_C, 0.0, schedule.ab2c_layout)
                with sb.for_loop('block_k_tile', block_k_tiles - 1) as k0:
                    with sb.for_loop('warp_k_tile', block_k // warp_k) as k1:
                        with sb.if_then(Equal(k1, 0)):
                            sb += self.copy(smem_A[k0 % 2, :, k1 + 1:], regs_A[(k1 + 1) % 2], schedule.a_s2r_layout)
                            sb += self.copy(smem_B[k0 % 2, k1 + 1:, :], regs_B[(k1 + 1) % 2], schedule.b_s2r_layout)
                            sb += self.transfer(dst=regs_A_ldg, src=gmem_A, layout=schedule.a_g2r_r2s_layout, f_src_index=lambda i, k: [block_m, i, k + k0 * block_k + first_k_tile], protect_src=True)
                            sb += self.transfer(dst=regs_B_ldg, src=gmem_B, layout=schedule.b_g2r_r2s_layout, f_src_index=lambda k, j: [block_n, k + k0 * block_k + first_k_tile, j], protect_src=True)
                            sb += self.mma(regs_A[k1 % 2], regs_B[k1 % 2], regs_C, schedule)
                        with sb.otherwise():
                            with sb.if_then(k1 < (block_k - 1)):
                                sb += self.copy(smem_A[k0 % 2, :, k1 + 1:], regs_A[(k1 + 1) % 2], schedule.a_s2r_layout)
                                sb += self.copy(smem_B[k0 % 2, k1 + 1:, :], regs_B[(k1 + 1) % 2], schedule.b_s2r_layout)
                                sb += self.mma(regs_A[k1 % 2], regs_B[k1 % 2], regs_C, schedule)
                            with sb.otherwise():
                                sb += self.copy(regs_A_ldg, smem_A[(k0 + 1) % 2], layout=schedule.a_g2r_r2s_layout)
                                sb += self.copy(regs_B_ldg, smem_B[(k0 + 1) % 2], layout=schedule.b_g2r_r2s_layout)
                                sb += syncthreads()
                                sb += self.copy(smem_A[(k0 + 1) % 2], regs_A[(k1 + 1) % 2], schedule.a_s2r_layout)
                                sb += self.copy(smem_B[(k0 + 1) % 2], regs_B[(k1 + 1) % 2], schedule.b_s2r_layout)
                                sb += self.mma(regs_A[k1 % 2], regs_B[k1 % 2], regs_C, schedule)
                with sb.let('block_k_tile', block_k_tiles - 1) as k0:
                    with sb.for_loop('warp_k_tile', block_k // warp_k) as k1:
                        with sb.if_then(k1 < block_k - 1):
                            sb += self.copy(smem_A[k0 % 2, :, k1 + 1:], regs_A[(k1 + 1) % 2], schedule.a_s2r_layout)
                            sb += self.copy(smem_B[k0 % 2, k1 + 1:, :], regs_B[(k1 + 1) % 2], schedule.b_s2r_layout)
                            sb += self.mma(regs_A[k1 % 2], regs_B[k1 % 2], regs_C, schedule)
                        with sb.otherwise():
                            sb += self.mma(regs_A[k1 % 2], regs_B[k1 % 2], regs_C, schedule)
                # regs -> gmem
                with sb.for_loop('i', schedule.c_r2s_outer_outer[0]) as i:
                    with sb.for_loop('j', schedule.c_r2s_outer_outer[1]) as j:
                        sb += syncthreads()
                        # sb += c_r2s(~regs_C_wb[i, j, 0, 0], smem_C)
                        sb += self.copy(regs_C_wb[i, j], smem_C, schedule.c_r2s_layout)
                        sb += syncthreads()
                        sb += self.transfer(dst=gmem_C_wb, src=smem_C, f_dst_index=lambda ii, jj: [block_m, block_n, i, j, ii, jj], layout=schedule.c_s2g_layout, protect_dst=True)
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

    def copy(self, src, dst, layout):
        sb = StmtBuilder()
        for indices in layout(thread_idx()):
            sb += BufferStoreStmt(dst, indices, src.__getitem__(indices))
        return sb.finish()

    def mma(self, a, b, c, schedule):
        layout = schedule.ab2c_layout
        sb = StmtBuilder()
        for i, j in layout(thread_idx()):
            for k in range(schedule.warp_k):
                sb += BufferStoreStmt(c, [i, j], c[i, j] + a[i, k] * b[k, j])
        return sb.finish()

    def transfer(self,
                 dst: Var,
                 src: Var,
                 layout: TaskLayout,
                 worker_idx: Var = thread_idx(),
                 f_dst_index: Callable[[Any], List[Expr]] = lambda *args: args,
                 f_src_index: Callable[[Any], List[Expr]] = lambda *args: args,
                 protect_src: bool = False,
                 protect_dst: bool = False,
                 src_predicate: Callable[[Any], Condition] = None,
                 src_default: Union[Expr, PyScalar] = 0.0,
                 ) -> Stmt:
        src_layout: DataLayout = get_tensor_layout(src)
        dst_layout: DataLayout = get_tensor_layout(dst)
        sb = StmtBuilder()
        for task_index in layout(worker_idx):
            src_index = f_src_index(*task_index)
            src_cond = convert(True)
            if protect_src:
                src_cond = And(src_cond, src_layout.within_bound(src_index))
            if src_predicate:
                src_cond = And(src_cond, src_predicate(*task_index))
            src_value = IfThenElse(src_cond, TensorElement(src, src_index), src_default)
            with contextlib.ExitStack() as stack:
                dst_index = f_dst_index(*task_index)
                assert len(src_index) == len(src_layout.shape)
                assert len(dst_index) == len(dst_layout.shape)
                if protect_dst:
                    stack.enter_context(sb.if_then(dst_layout.within_bound(*dst_index)))
                sb += BufferStoreStmt(dst, dst_index, src_value)
        return sb.finish()
