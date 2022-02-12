from typing import Mapping, List, Union

from hidet.implement.implementer import Implementer, register_impl, NotSupportedError
from hidet.implement.search_space import ProductSpace, AtomSpace, SpaceChoice
from hidet.ir.builders import TaskBuilder, FunctionBuilder, StmtBuilder
from hidet.ir.dialects.compute import TensorInput, TensorCompute, compute, ReduceCompute, reduce_sum
from hidet.ir.dialects.lowlevel import Address, TensorPointerType, Cast, PointerType
from hidet.ir.dialects.pattern import TaskPattern, any_const_int
from hidet.ir.functors import simplify, simplify_to_int
from hidet.ir.expr import Expr, Call, TensorElement, var, tensor_var, convert, Var
from hidet.ir.func import IRModule
from hidet.ir.layout import TaskLayout, row_major_layout, full_layout
from hidet.ir.node import Node
from hidet.ir.primitives import syncthreads, thread_idx
from hidet.ir.stmt import LetStmt, ForStmt, AssignStmt
from hidet.ir.task import Task, ThreadBlock, Warp
from hidet.ir.type import scalar_type, TensorType, Scope, DataLayout, StridesLayout
from hidet.implement.common import transfer_task, init_task


class MatmulSetting:
    def __init__(self,
                 block_k=8,
                 warp_k=1,
                 block_warps=(4, 2),
                 outer=(2, 2),
                 atom_layout=TaskLayout(num_workers=32,
                                        task_shape=(4, 8),
                                        worker2task=lambda w: [(w // 16 * 2 + w % 2, w // 2 % 8)],
                                        task2worker=lambda i, j: i // 2 * 16 + i % 2 + j * 2),
                 inner=(4, 4)):
        self.block_k = block_k
        self.warp_k = warp_k
        self.block_warps = block_warps
        self.outer = outer
        self.atom_layout = atom_layout
        self.inner = inner
        # dependent variables
        warp_m = outer[0] * atom_layout.task_shape[0] * inner[0]
        warp_n = outer[1] * atom_layout.task_shape[1] * inner[1]
        block_m = block_warps[0] * warp_m
        block_n = block_warps[1] * warp_n
        atom_shape = atom_layout.task_shape

        warp_size = 32
        num_warps = block_warps[0] * block_warps[1]
        block_size = num_warps * warp_size
        assert block_k % warp_k == 0 and (block_k // warp_k) % 2 == 0  # double buffering index requirement
        if warp_size % block_k != 0 and block_k % warp_size == 0:  # this is another case we might handle
            raise NotImplementedError()
        assert warp_size % block_k == 0  # gmem -> regs requirement. block_k candidates: 2, 4, 8, 16, 32
        assert block_m % (block_size // block_k) == 0  # A: gmem -> regs alignment requirement
        assert block_n % (block_size // block_k) == 0  # B: gmem -> regs alignment requirement
        if warp_k > 1:
            raise NotImplementedError()

        # task layouts
        self.block_layout = row_major_layout(*block_warps)
        self.warp_layout = full_layout(*outer) * atom_layout * full_layout(*inner)
        self.a_g2r_r2s_layout = row_major_layout(block_size // block_k, block_k) * full_layout(block_m // (block_size // block_k), 1)
        self.b_g2r_r2s_layout = full_layout(1, block_n // (block_size // block_k)) * row_major_layout(block_k, block_size // block_k)
        self.a_s2r_layout: TaskLayout = (self.block_layout * full_layout(outer[0], 1) * atom_layout * full_layout(inner[0], 1)).projection({1: 0})
        self.b_s2r_layout: TaskLayout = (self.block_layout * full_layout(1, outer[1]) * atom_layout * full_layout(1, inner[1])).projection({0: 0})
        self.ab2c_layout = self.block_layout * self.warp_layout
        self.c_r2s_outer_inner = self.get_c_write_back_inner()
        self.c_r2s_s2g_layout = self.block_layout * full_layout(*self.c_r2s_outer_inner) * atom_layout * full_layout(*inner)
        self.c_r2s_outer_outer = [outer[0] // self.c_r2s_outer_inner[0], outer[1] // self.c_r2s_outer_inner[1]]

        self.wb_warp_shape = [atom_shape[0] * inner[0] * self.c_r2s_outer_inner[0], atom_shape[1] * inner[1] * self.c_r2s_outer_inner[1]]

        # regs data layout
        self.regs_a_layout = DataLayout.local((block_warps[0], 1)) * StridesLayout.row_major((outer[0], 1)) * DataLayout.local((atom_shape[0], 1)) * StridesLayout.row_major((inner[0], 1))
        self.regs_b_layout = DataLayout.local((1, block_warps[1])) * StridesLayout.row_major((1, outer[1])) * DataLayout.local((1, atom_shape[1])) * StridesLayout.row_major((1, inner[1]))
        self.regs_c_layout = DataLayout.local(block_warps) * StridesLayout.row_major(outer) * DataLayout.local(atom_shape) * StridesLayout.row_major(inner)
        self.regs_a_ldg_layout = DataLayout(size=block_m * block_k // block_size, shape=(block_m, block_k), global2local=(lambda i, j: i % (block_m // (block_size // block_k))))
        self.regs_b_ldg_layout = DataLayout(size=block_k * block_n // block_size, shape=(block_k, block_n), global2local=(lambda i, j: j // (block_size // block_k)))

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
                smem_size = (block_m + block_n) * self.block_k * 2
                if regs_size > smem_size:
                    continue
                if found_regs_size == 0 or found_regs_size <= regs_size:
                    found_regs_size = regs_size
                    c_r2s_outer_inner = [a, b]
        if found_regs_size == 0:
            raise NotSupportedError('can not use smem to help write c.')
        return c_r2s_outer_inner


matmul_settings = [MatmulSetting()]
# matmul_settings = [MatmulSetting(
#     block_k=2,
#     block_warps=[1, 1],
#     outer=[1, 1],
#     inner=[4, 2]
# )]


@register_impl('cuda_block_static_matmul_soft_pipe_ldg_wb_implementer')
class CudaBlockStaticMatmulSoftPipeLdgWbImplementer(Implementer):
    def __init__(self):
        # const definition
        self.block_size = any_const_int()
        self.task_m = any_const_int()
        self.task_n = any_const_int()
        self.task_k = any_const_int()

        # inputs
        A = TensorInput('A', dtype=scalar_type('float32'))
        B = TensorInput('B', dtype=scalar_type('float32'))

        # compute
        i, j, k = var('i'), var('j'), var('k')
        computation = TensorCompute(name='C',
                                    shape=[self.task_m, self.task_n],
                                    axes=[i, j],
                                    value=ReduceCompute(
                                        value=A[i, k] * B[k, j],
                                        shape=[self.task_k],
                                        axis=k,
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
            worker=ThreadBlock(block_dim=self.block_size)
        )

    def priority(self) -> int:
        return 1

    def task_pattern(self) -> TaskPattern:
        return self.pattern

    def implement(self, task: Task, match: Mapping[Node, Node]) -> IRModule:
        # search space
        block_size = int(match[self.block_size])
        assert block_size % 32 == 0
        space = ProductSpace(
            name='cuda_block_static_matmul_implementer',
            sub_spaces=[
                AtomSpace('setting', matmul_settings)
            ])

        space_size = len(space)
        ir_module = IRModule(task=task)
        for i in range(space_size):
            choice = space[i]
            try:
                sub_ir_module = self.implement_for_choice(task, match, choice)
            except NotSupportedError as e:
                continue
            else:
                ir_module.include(sub_ir_module)
        if len(ir_module.functions) == 0:
            raise NotSupportedError('Can not find a setting to implement task {}'.format(task.name))
        return ir_module

    def implement_for_choice(self, task: Task, match: Mapping[Node, Node], choice: SpaceChoice) -> IRModule:
        ir_module = IRModule()

        # task-related constants
        block_size = int(match[self.block_size])
        task_m = int(match[self.task_m])
        task_n = int(match[self.task_n])
        task_k = int(match[self.task_k])

        # space-related constants
        setting: MatmulSetting = choice.setting.value
        block_layout: TaskLayout = setting.block_layout
        warp_layout: TaskLayout = setting.warp_layout

        block_m, block_n = block_layout.task_shape
        block_k = setting.block_k

        warp_m, warp_n = warp_layout.task_shape
        warp_k = setting.warp_k

        wb_warp_shape = setting.wb_warp_shape

        self.check(block_n * warp_n == task_n)
        self.check(block_m * warp_m == task_m)
        self.check(block_size == block_layout.num_workers * warp_layout.num_workers)
        self.check(task_k % (block_k * warp_k) == 0)  # currently, we only implement perfect fit case

        # other constants
        self.check(block_size % 32 == 0)

        # task-related variables
        A_type: TensorType = match[self.A_type]
        B_type: TensorType = match[self.B_type]
        C_type: TensorType = match[self.C_type]

        # todo: correct the type, will fix the upstream implementor later. Now we fix it here
        C_layout: StridesLayout = C_type.layout
        C_type = TensorType(scope=C_type.scope, dtype=C_type.scalar_type, layout=StridesLayout(shape=(task_m, task_n), strides=C_layout.strides))

        A_dtype = A_type.scalar_type
        B_dtype = B_type.scalar_type
        C_dtype = C_type.scalar_type

        # declare inputs and outputs and their types shared by all subtasks
        smem_A_type = TensorType(scope='shared', dtype=A_dtype, shape=[task_m, block_k * warp_k], layout=[1, task_m])  # column major, TODO: add to search space
        smem_B_type = TensorType(scope='shared', dtype=B_dtype, shape=[block_k * warp_k, task_n], layout=[task_n, 1])  # row major
        regs_A_type = TensorType(scope='register', dtype=A_dtype, layout=setting.regs_a_layout)
        regs_B_type = TensorType(scope='register', dtype=B_dtype, layout=setting.regs_b_layout)
        regs_C_type = TensorType(scope='register', dtype=C_dtype, layout=setting.regs_c_layout)
        regs_A_ldg_type = TensorType(scope='register', dtype=A_dtype, layout=setting.regs_a_ldg_layout)
        regs_B_ldg_type = TensorType(scope='register', dtype=B_dtype, layout=setting.regs_b_ldg_layout)
        #                                                                 (M, N) ->     (BM, WM, BN, WN) ->           (BM, OM, IM, BN, ON, IN) ->                       (OM, ON, BM * IM, BN * IN)
        gmem_C_wb_type = TensorType(scope='global', dtype=C_dtype, layout=C_type.layout.split({0: warp_m, 1: warp_n}).split({1: wb_warp_shape[0], 3: wb_warp_shape[1]}).fuse([1, 4, [0, 2], [3, 5]]))
        smem_C_wb_type = TensorType(scope='shared', dtype=C_dtype, layout=StridesLayout.row_major(setting.c_r2s_s2g_layout.task_shape))
        regs_C_wb_type = TensorType(scope='register', dtype=C_dtype, layout=regs_C_type.layout.split({0: warp_m, 1: warp_n}).split({1: wb_warp_shape[0], 3: wb_warp_shape[1]}).fuse([1, 4, [0, 2], [3, 5]]))

        # define subtasks
        c_init = init_task(f'{task.name}.c.init', dst_type=regs_C_type, init_value=convert(0.0), worker=ThreadBlock(task_layout=setting.ab2c_layout), parent_module=ir_module)
        a_g2r = transfer_task(f'{task.name}.a.g2r.block', src_type=A_type, dst_type=regs_A_ldg_type, worker=ThreadBlock(task_layout=setting.a_g2r_r2s_layout), parent_module=ir_module)
        b_g2r = transfer_task(f'{task.name}.b.g2r.block', src_type=B_type, dst_type=regs_B_ldg_type, worker=ThreadBlock(task_layout=setting.b_g2r_r2s_layout), parent_module=ir_module)
        a_r2s = transfer_task(f'{task.name}.a_r2s.block', src_type=regs_A_ldg_type, dst_type=smem_A_type, worker=ThreadBlock(task_layout=setting.a_g2r_r2s_layout), parent_module=ir_module)
        b_r2s = transfer_task(f'{task.name}.b_r2s.block', src_type=regs_B_ldg_type, dst_type=smem_B_type, worker=ThreadBlock(task_layout=setting.b_g2r_r2s_layout), parent_module=ir_module)
        a_s2r = transfer_task(f'{task.name}.a.s2r.block', src_type=smem_A_type, dst_type=regs_A_type, worker=ThreadBlock(task_layout=setting.a_s2r_layout), parent_module=ir_module)
        b_s2r = transfer_task(f'{task.name}.b.s2r.block', src_type=smem_B_type, dst_type=regs_B_type, worker=ThreadBlock(task_layout=setting.b_s2r_layout), parent_module=ir_module)
        c_r2s = transfer_task(f'{task.name}.c.r2s.block', src_type=regs_C_wb_type.slice_out(dims=[0, 1]), dst_type=smem_C_wb_type, worker=ThreadBlock(task_layout=setting.c_r2s_s2g_layout), parent_module=ir_module)
        c_s2g = transfer_task(f'{task.name}.c.s2g.block', src_type=smem_C_wb_type, dst_type=gmem_C_wb_type.slice_out(dims=[0, 1]), worker=ThreadBlock(task_layout=setting.c_r2s_s2g_layout), parent_module=ir_module)

        with TaskBuilder(f'{task.name}.compute.warp', Warp(setting.ab2c_layout), ir_module) as ab2c:
            regs_A_input = TensorInput('regs_A', A_dtype)
            regs_B_input = TensorInput('regs_B', B_dtype)
            axis_k = var('k')
            fcompute = lambda i, j: reduce_sum(regs_A_input[i, axis_k] * regs_B_input[axis_k, j], axis=axis_k, shape=[warp_k])
            ab2c_cmpt = compute('regs_C', shape=[warp_m, warp_n], fcompute=fcompute)
            ab2c.set_computation(ab2c_cmpt)
            ab2c.append_param(regs_A_input, regs_A_type)
            ab2c.append_param(regs_B_input, regs_B_type)
            ab2c.append_param(ab2c_cmpt, regs_C_type)

        # define function
        with FunctionBuilder(task.name) as fb:
            """
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
            sb = StmtBuilder()

            # declare params
            gmem_A = Var('A', A_type)
            gmem_B = Var('B', B_type)
            gmem_C = Var('C', C_type)
            fb.extend_params([gmem_A, gmem_B, gmem_C])

            # declare local variables
            smem_A_tp = TensorPointerType('shared', A_dtype, layout=StridesLayout.row_major([2]) + smem_A_type.layout)
            smem_B_tp = TensorPointerType('shared', B_dtype, layout=StridesLayout.row_major([2]) + smem_B_type.layout)
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
            regs_A = Var('regs_A', TensorType('register', A_dtype, [2] + regs_A_type.shape, StridesLayout.row_major([2]) + regs_A_type.layout))
            regs_B = Var('regs_B', TensorType('register', B_dtype, [2] + regs_B_type.shape, StridesLayout.row_major([2]) + regs_B_type.layout))
            regs_C = Var('regs_C', regs_C_type)
            regs_A_ldg = Var('regs_A_ldg', regs_A_ldg_type)
            regs_B_ldg = Var('regs_B_ldg', regs_B_ldg_type)
            regs_C_wb = Var('regs_C_wb', TensorPointerType('register', C_dtype, layout=regs_C_wb_type.layout))
            sb += AssignStmt(regs_C_wb, regs_C)
            fb.extend_local_vars([regs_A, regs_B, regs_C, regs_A_ldg, regs_B_ldg, regs_C_wb])

            # function body
            # init regs c
            sb += c_init(regs_C)
            # transfer first tile
            sb += a_g2r(~gmem_A[0, 0], regs_A_ldg)
            sb += a_r2s(regs_A_ldg, smem_A)
            sb += b_g2r(~gmem_B[0, 0], regs_B_ldg)
            sb += b_r2s(regs_B_ldg, smem_B)
            sb += syncthreads()
            sb += a_s2r(~smem_A[0, 0, 0], ~regs_A[0, 0, 0])
            sb += b_s2r(~smem_B[0, 0, 0], ~regs_B[0, 0, 0])
            sb += syncthreads()
            with sb.for_loop('block_k_tile', task_k // (block_k * warp_k) - 1) as k0:
                with sb.for_loop('warp_k_tile', block_k) as k1:
                    with sb.if_then(k1 == 0):
                        sb += a_s2r(~smem_A[k0 % 2, 0, k1 + 1], ~regs_A[(k1 + 1) % 2, 0, 0])
                        sb += b_s2r(~smem_B[k0 % 2, k1 + 1, 0], ~regs_B[(k1 + 1) % 2, 0, 0])
                        sb += a_g2r(~gmem_A[0, (k0 + 1) * (block_k * warp_k)], regs_A_ldg)
                        sb += b_g2r(~gmem_B[(k0 + 1) * (block_k * warp_k), 0], regs_B_ldg)
                        sb += ab2c(~regs_A[k1 % 2, 0, 0], ~regs_B[k1 % 2, 0, 0], regs_C)
                    with sb.otherwise():
                        with sb.if_then(k1 < (block_k * warp_k - 1)):
                            sb += a_s2r(~smem_A[k0 % 2, 0, k1 + 1], ~regs_A[(k1 + 1) % 2, 0, 0])
                            sb += b_s2r(~smem_B[k0 % 2, k1 + 1, 0], ~regs_B[(k1 + 1) % 2, 0, 0])
                            sb += ab2c(~regs_A[k1 % 2, 0, 0], ~regs_B[k1 % 2, 0, 0], regs_C)
                        with sb.otherwise():
                            sb += a_r2s(regs_A_ldg, ~smem_A[(k0 + 1) % 2, 0, 0])
                            sb += b_r2s(regs_B_ldg, ~smem_B[(k0 + 1) % 2, 0, 0])
                            sb += syncthreads()
                            sb += a_s2r(~smem_A[(k0 + 1) % 2, 0, 0], ~regs_A[(k1 + 1) % 2, 0, 0])
                            sb += b_s2r(~smem_B[(k0 + 1) % 2, 0, 0], ~regs_B[(k1 + 1) % 2, 0, 0])
                            sb += ab2c(~regs_A[k1 % 2, 0, 0], ~regs_B[k1 % 2, 0, 0], regs_C)
            with sb.let('block_k_tile', task_k // (block_k * warp_k) - 1) as k0:
                with sb.for_loop('warp_k_tile', block_k) as k1:
                    with sb.if_then(k1 < block_k * warp_k - 1):
                        sb += a_s2r(~smem_A[k0 % 2, 0, k1 + 1], ~regs_A[(k1 + 1) % 2, 0, 0])
                        sb += b_s2r(~smem_B[k0 % 2, k1 + 1, 0], ~regs_B[(k1 + 1) % 2, 0, 0])
                        sb += ab2c(~regs_A[k1 % 2, 0, 0], ~regs_B[k1 % 2, 0, 0], regs_C)
                    with sb.otherwise():
                        sb += ab2c(~regs_A[k1 % 2, 0, 0], ~regs_B[k1 % 2, 0, 0], regs_C)
            # regs -> gmem
            with sb.for_loop('i', setting.c_r2s_outer_outer[0]) as i:
                with sb.for_loop('j', setting.c_r2s_outer_outer[1]) as j:
                    sb += syncthreads()
                    sb += c_r2s(~regs_C_wb[i, j, 0, 0], smem_C)
                    sb += syncthreads()
                    sb += c_s2g(smem_C, ~gmem_C_wb[i, j, 0, 0])

            # set body
            fb.set_body(sb.finish())

            # attrs
            fb.extend_attrs({'worker': task.worker, 'label': 'matmul128x128x8'})

        func = fb.get()
        ir_module.add(func.name, func)
        return ir_module
