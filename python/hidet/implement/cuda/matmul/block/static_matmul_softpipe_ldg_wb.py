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
        self.a_s2r_layout: TaskLayout = (self.block_layout * full_layout(outer[0], 1) * atom_layout * full_layout(inner[0], 1)).projection(None, 0)
        self.b_s2r_layout: TaskLayout = (self.block_layout * full_layout(1, outer[1]) * atom_layout * full_layout(1, inner[1])).projection(0, None)
        self.ab2c_layout = self.block_layout * self.warp_layout
        # c_r2s_outer_inner = self.get_c_write_back_inner()
        # self.c_r2s_s2g_layout = self.block_layout * full_layout(c_r2s_outer_inner) * atom_layout * full_layout(*inner)
        # self.c_r2s_outer_outer = [outer[0] // c_r2s_outer_inner[0], outer[1] // c_r2s_outer_inner[1]]
        self.c_r2g_layout: TaskLayout = self.block_layout * self.warp_layout

        # regs data layout
        middle = [atom_shape[0] * inner[0], atom_shape[1] * inner[1]]
        outer_layout = DataLayout.local_layout(block_warps)
        self.regs_a_layout = outer_layout * DataLayout(size=outer[0] * inner[0], shape=(warp_m, 1), global2local=(lambda i, j: i % inner[0] + (i // middle[0]) * inner[0]))
        self.regs_b_layout = outer_layout * DataLayout(size=outer[1] * inner[1], shape=(1, warp_n), global2local=(lambda i, j: j % inner[1] + (j // middle[1]) * inner[1]))
        self.regs_c_layout = outer_layout * DataLayout(size=outer[0] * inner[0] * outer[1] * inner[1], shape=(warp_m, warp_n),
                                                       global2local=(lambda i, j: ((i // middle[0] * inner[0] + i % inner[0]) * outer[1] * inner[1] + (j // middle[1] * inner[1] + j % inner[1]))))
        self.regs_a_ldg_layout = DataLayout(size=block_m * block_k // block_size, shape=(block_m, block_k), global2local=(lambda i, j: i % (block_m // (block_size // block_k))))
        self.regs_b_ldg_layout = DataLayout(size=block_k * block_n // block_size, shape=(block_k, block_n), global2local=(lambda i, j: j // (block_size // block_k)))

    def get_c_write_back_inner(self):
        block_size = self.block_warps[0] * self.block_warps[1] * 32
        block_m = self.block_warps[0] * self.atom_layout.task_shape[0]
        block_n = self.block_warps[1] * self.atom_layout.task_shape[1]
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


@register_impl('cuda_block_static_matmul_soft_pipe_ldg_wb_implementer')
class CudaBlockStaticMatmulSoftPipeLdgWbImplementer(Implementer):
    def __init__(self):
        # const definition
        self.block_size = any_const_int()
        self.task_m = any_const_int()
        self.task_n = any_const_int()
        self.task_k = any_const_int()

        # inputs
        self.A = TensorInput('A', dtype=scalar_type('float32'), shape=None)
        self.B = TensorInput('B', dtype=scalar_type('float32'), shape=None)

        # compute
        self.axis_i = var('i')
        self.axis_j = var('j')
        self.axis_k = var('k')
        self.value = self.A[self.axis_i, self.axis_k] * self.B[self.axis_k, self.axis_j]
        self.reduce = ReduceCompute(value=self.value,
                                    shape=[self.task_k],
                                    axis=self.axis_k,
                                    reduce_type=None)
        self.computation = TensorCompute(name='C',
                                         shape=[self.task_m, self.task_n],
                                         axes=[self.axis_i, self.axis_j],
                                         value=self.reduce)

        # inputs and output types
        self.A_layout = DataLayout()
        self.B_layout = DataLayout()
        self.C_layout = DataLayout()
        self.A_type = TensorType(Scope('global'), scalar_type('float32'), shape=None, layout=self.A_layout)
        self.B_type = TensorType(Scope('global'), scalar_type('float32'), shape=None, layout=self.B_layout)
        self.C_type = TensorType(Scope('global'), scalar_type('float32'), shape=None, layout=self.C_layout)

        # pattern
        self.pattern = TaskPattern(
            compute_pattern=self.computation,
            required_params=[self.A, self.B, self.computation],
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

        self.check(block_n * warp_n == task_n)
        self.check(block_m * warp_m == task_m)
        self.check(block_size == block_layout.num_workers * warp_layout.num_workers)
        self.check(task_k % (block_k * warp_k) == 0)  # currently, we only implement perfect fit case

        # other constants
        self.check(block_size % 32 == 0)

        # task-related variables
        A: TensorInput = match[self.A]
        B: TensorInput = match[self.B]
        A_type: TensorType = match[self.A_type]
        B_type: TensorType = match[self.B_type]
        C_type: TensorType = match[self.C_type]
        A_dtype = A_type.scalar_type
        B_dtype = B_type.scalar_type
        C_dtype = C_type.scalar_type

        # declare inputs and outputs and their types shared by all subtasks
        smem_A_type = TensorType(scope='shared', dtype=A.dtype, shape=[task_m, block_k * warp_k], layout=[1, task_m])  # column major, TODO: add to search space
        smem_B_type = TensorType(scope='shared', dtype=B.dtype, shape=[block_k * warp_k, task_n], layout=[task_n, 1])  # row major

        regs_A_type = TensorType(scope='register', dtype=A_dtype, shape=[warp_m, warp_k], layout=setting.regs_a_layout)
        regs_B_type = TensorType(scope='register', dtype=B_dtype, shape=[warp_k, warp_n], layout=setting.regs_b_layout)
        regs_C_type = TensorType(scope='register', dtype=C_dtype, shape=[warp_m, warp_n], layout=setting.regs_c_layout)

        regs_A_ldg_type = TensorType(scope='register', dtype=A_dtype, shape=[block_m, block_k * warp_k], layout=setting.regs_a_ldg_layout)
        regs_B_ldg_type = TensorType(scope='register', dtype=B_dtype, shape=[block_k * warp_k, block_n], layout=setting.regs_b_ldg_layout)

        # define subtasks
        with TaskBuilder(f'{task.name}.c.init.warp', Warp(setting.ab2c_layout), ir_module, 'cuda_warp_fill_value_implementer') as c_init:
            c_init_cmpt = compute('regs_c', shape=[warp_m, warp_n], fcompute=(lambda i, j: convert(0.0)))
            c_init.set_computation(c_init_cmpt)
            c_init.append_param(c_init_cmpt, regs_C_type)

        with TaskBuilder(f'{task.name}.a.g2r.block', ThreadBlock(block_size, setting.a_g2r_r2s_layout), ir_module) as a_g2r:
            gmem_frag_A = TensorInput('gmem_frag_A', A_dtype)
            gmem_frag_A_type = TensorType('global', A_dtype, layout=A_type.layout)
            a_g2r_cmpt = compute('regs_a_ldg', shape=[task_m, block_k * warp_k], fcompute=lambda i, j: gmem_frag_A[i, j])
            a_g2r.set_computation(a_g2r_cmpt)
            a_g2r.append_param(gmem_frag_A, gmem_frag_A_type)
            a_g2r.append_param(a_g2r_cmpt, regs_A_ldg_type)

        with TaskBuilder(f'{task.name}.b.g2r.block', ThreadBlock(block_size, setting.b_g2r_r2s_layout), ir_module) as b_g2r:
            gmem_frag_B = TensorInput('gmem_frag_B', B_dtype)
            gmem_frag_B_type = TensorType('global', B_dtype, layout=B_type.layout)
            b_g2r_cmpt = compute('regs_a_ldg', shape=[block_k * warp_k, task_n], fcompute=lambda i, j: gmem_frag_B[i, j])
            b_g2r.set_computation(b_g2r_cmpt)
            b_g2r.append_param(gmem_frag_B, gmem_frag_B_type)
            b_g2r.append_param(b_g2r_cmpt, regs_B_ldg_type)

        with TaskBuilder(f'{task.name}.a.r2s.block', ThreadBlock(block_size, setting.a_g2r_r2s_layout), ir_module) as a_r2s:
            regs_A_ldg = TensorInput('regs_A_ldg', A_dtype)
            a_r2s_cmpt = compute('smem_A', shape=[task_m, block_k * warp_k], fcompute=lambda i, j: regs_A_ldg[i, j])
            a_r2s.set_computation(a_r2s_cmpt)
            a_r2s.append_param(regs_A_ldg, regs_A_ldg_type)
            a_r2s.append_param(a_r2s_cmpt, smem_A_type)

        with TaskBuilder(f'{task.name}.b.r2s.block', ThreadBlock(block_size, setting.b_g2r_r2s_layout), ir_module) as b_r2s:
            regs_B_ldg = TensorInput('regs_B_ldg', B_dtype)
            b_r2s_cmpt = compute('smem_B', shape=[block_k * warp_k, task_n], fcompute=lambda i, j: regs_B_ldg[i, j])
            b_r2s.set_computation(b_r2s_cmpt)
            b_r2s.append_param(regs_B_ldg, regs_B_ldg_type)
            b_r2s.append_param(b_r2s_cmpt, smem_B_type)

        with TaskBuilder(f'{task.name}.a.s2r.block', ThreadBlock(block_size, setting.a_s2r_layout), ir_module) as a_s2r:
            smem_frag_A = TensorInput('smem_frag_A', A.dtype)
            smem_frag_A_type = TensorType('shared', A.dtype, layout=smem_A_type.layout)
            a_s2r_cmpt = compute('regs_A', shape=setting.a_s2r_layout.task_shape, fcompute=lambda i, j: smem_frag_A[i, j])
            a_s2r.set_computation(a_s2r_cmpt)
            a_s2r.append_param(smem_frag_A, smem_frag_A_type)
            a_s2r.append_param(a_s2r_cmpt, regs_A_type)

        with TaskBuilder(f'{task.name}.b.s2r.block', ThreadBlock(block_size, setting.b_s2r_layout), ir_module) as b_s2r:
            smem_frag_B = TensorInput('smem_frag_B', B.dtype)
            smem_frag_B_type = TensorType('shared', B.dtype, layout=smem_B_type.layout)
            b_s2r_cmpt = compute('regs_B', shape=setting.b_s2r_layout.task_shape, fcompute=lambda i, j: smem_frag_B[i, j])
            b_s2r.set_computation(b_s2r_cmpt)
            b_s2r.append_param(smem_frag_B, smem_frag_B_type)
            b_s2r.append_param(b_s2r_cmpt, regs_B_type)

        with TaskBuilder(f'{task.name}.compute.warp', Warp(setting.ab2c_layout), ir_module) as ab2c:
            regs_A_input = TensorInput('regs_A', A.dtype)
            regs_B_input = TensorInput('regs_B', B.dtype)
            axis_k = var('k')
            fcompute = lambda i, j: reduce_sum(regs_A_input[i, axis_k] * regs_B_input[axis_k, j], axis=axis_k, shape=[warp_k])
            ab2c_cmpt = compute('regs_C', shape=[warp_m, warp_n], fcompute=fcompute)
            ab2c.set_computation(ab2c_cmpt)
            ab2c.append_param(regs_A_input, regs_A_type)
            ab2c.append_param(regs_B_input, regs_B_type)
            ab2c.append_param(ab2c_cmpt, regs_C_type)

        with TaskBuilder(f'{task.name}.r2g.warp', Warp(setting.c_r2g_layout), ir_module) as c_r2g:
            regs_C_input = TensorInput('regs_C', C_dtype)
            c_r2g_cmpt = compute('gmem_C', shape=[warp_m, warp_n], fcompute=lambda i, j: regs_C_input[i, j])
            c_r2g.set_computation(c_r2g_cmpt)
            c_r2g.append_param(regs_C_input, regs_C_type)
            c_r2g.append_param(c_r2g_cmpt, C_type)

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

            # declare A, B shared memory
            smem_A_tp = TensorPointerType('shared', A_dtype, [2] + smem_A_type.shape, StridesLayout.row_major([2]) + smem_A_type.layout)
            smem_B_tp = TensorPointerType('shared', B_dtype, [2] + smem_B_type.shape, StridesLayout.row_major([2]) + smem_B_type.layout)
            # smem_C_tp = TensorPointerType('shared', C_dtype)
            smem_A = Var('smem_A', smem_A_tp)
            smem_B = Var('smem_B', smem_B_tp)
            # smem_C = Var('smem_C', smem_C_tp)
            smem_A_bytes = simplify_to_int(smem_A_tp.tensor_type.storage_bytes())
            smem_B_bytes = simplify_to_int(smem_B_tp.tensor_type.storage_bytes())
            # smem_C_bytes = simplify_to_int(smem_C_tp.tensor_type.storage_bytes())
            # smem_storage = Var('smem_storage', TensorType('shared', dtype='uint8', shape=[max(smem_A_bytes + smem_B_bytes, smem_C_bytes)], layout=[1]))
            smem_storage = Var('smem_storage', TensorType('shared', dtype='uint8', shape=[max(smem_A_bytes + smem_B_bytes, 0)], layout=[1]))
            # fb.extend_local_vars([smem_A, smem_B, smem_C, smem_storage])
            fb.extend_local_vars([smem_A, smem_B, smem_storage])
            sb += AssignStmt(smem_A, Cast(Address(smem_storage[0]), PointerType(A_dtype)))
            sb += AssignStmt(smem_B, Cast(Address(smem_storage[smem_A_bytes]), PointerType(B_dtype)))
            # sb += AssignStmt(smem_C, Address(smem_storage[0]))

            # declare A, B, C registers
            regs_A = Var('regs_A', TensorType('register', A_dtype, [2] + regs_A_type.shape, StridesLayout.row_major([2]) + regs_A_type.layout))
            regs_B = Var('regs_B', TensorType('register', B_dtype, [2] + regs_B_type.shape, StridesLayout.row_major([2]) + regs_B_type.layout))
            regs_C = Var('regs_C', regs_C_type)
            regs_A_ldg = Var('regs_A_ldg', regs_A_ldg_type)
            regs_B_ldg = Var('regs_B_ldg', regs_B_ldg_type)
            fb.extend_local_vars([regs_A, regs_B, regs_C, regs_A_ldg, regs_B_ldg])

            # function body
            # init regs c
            sb += c_init(regs_C)
            with sb.let('warp_id', thread_idx() // 32) as warp_idx:
                # transfer first tile
                sb += a_g2r(Address(gmem_A[0, 0]), regs_A_ldg)
                sb += a_r2s(regs_A_ldg, smem_A)
                sb += b_g2r(Address(gmem_B[0, 0]), regs_B_ldg)
                sb += b_r2s(regs_B_ldg, smem_B)
                sb += syncthreads()
                sb += a_s2r(Address(smem_A[0, 0, 0]), Address(regs_A[0, 0, 0]))
                sb += b_s2r(Address(smem_B[0, 0, 0]), Address(regs_B[0, 0, 0]))
                sb += syncthreads()
                with sb.for_loop('block_k_tile', task_k // (block_k * warp_k) - 1) as k0:
                    with sb.for_loop('warp_k_tile', block_k) as k1:
                        with sb.if_then(k1 == 0):
                            assert regs_A_type.layout.serialize(0, 0) == 0, "global index with only 0 must be mapped to 0 in local array"
                            sb += a_s2r(Address(smem_A[k0 % 2, 0, 0]), Address(regs_A[(k1 + 1) % 2, 0, 0]))
                            sb += b_s2r(Address(smem_B[k0 % 2, k1 + 1, 0]), Address(regs_B[(k1 + 1) % 2, 0, 0]))
                            sb += a_g2r(Address(gmem_A[0, (k0 + 1) * (block_k * warp_k)]), regs_A_ldg)
                            sb += b_g2r(Address(gmem_B[(k0 + 1) * (block_k * warp_k), 0]), regs_B_ldg)
                            sb += ab2c(Address(regs_A[k1 % 2, 0, 0]), Address(regs_B[k1 % 2, 0, 0]), regs_C)
                        with sb.otherwise():
                            with sb.if_then(k1 < (block_k * warp_k - 1)):
                                sb += a_s2r(Address(smem_A[k0 % 2, 0, k1 + 1]), Address(regs_A[(k1 + 1) % 2, 0, 0]))
                                sb += b_s2r(Address(smem_B[k0 % 2, k1 + 1, 0]), Address(regs_B[(k1 + 1) % 2, 0, 0]))
                                sb += ab2c(Address(regs_A[k1 % 2, 0, 0]), Address(regs_B[k1 % 2, 0, 0]), regs_C)
                            with sb.otherwise():
                                sb += a_r2s(regs_A_ldg, Address(smem_A[(k0 + 1) % 2, 0, 0]))
                                sb += b_r2s(regs_B_ldg, Address(smem_B[(k0 + 1) % 2, 0, 0]))
                                sb += syncthreads()
                                sb += a_s2r(Address(smem_A[(k0 + 1) % 2, 0, 0]), Address(regs_A[(k1 + 1) % 2, 0, 0]))
                                sb += b_s2r(Address(smem_B[(k0 + 1) % 2, 0, 0]), Address(regs_B[(k1 + 1) % 2, 0, 0]))
                                sb += ab2c(Address(regs_A[k1 % 2, 0, 0]), Address(regs_B[k1 % 2, 0, 0]), regs_C)
                with sb.let('block_k_tile', task_k // (block_k * warp_k) - 1) as k0:
                    with sb.for_loop('warp_k_tile', block_k) as k1:
                        with sb.if_then(k1 < block_k * warp_k - 1):
                            sb += a_s2r(Address(smem_A[k0 % 2, 0, k1 + 1]), Address(regs_A[(k1 + 1) % 2, 0, 0]))
                            sb += b_s2r(Address(smem_B[k0 % 2, k1 + 1, 0]), Address(regs_B[(k1 + 1) % 2, 0, 0]))
                            sb += ab2c(Address(regs_A[k1 % 2, 0, 0]), Address(regs_B[k1 % 2, 0, 0]), regs_C)
                        with sb.otherwise():
                            sb += ab2c(Address(regs_A[k1 % 2, 0, 0]), Address(regs_B[k1 % 2, 0, 0]), regs_C)
                sb += syncthreads()
                # regs -> gmem
                sb += c_r2g(regs_C, Address(gmem_C[warp_idx // 2 * 32, warp_idx % 2 * 64]))
            fb.set_body(sb.finish())

            # attrs
            fb.extend_attrs({'worker': task.worker, 'label': 'matmul128x128x8'})

        func = fb.get()
        ir_module.add(func.name, func)
        return ir_module
