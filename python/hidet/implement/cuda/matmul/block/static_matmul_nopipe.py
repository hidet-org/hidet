from typing import Mapping

from hidet.implement.cuda.layout import TaskLayout, row_major_layout, full_layout
from hidet.implement.implementer import Implementer, register_impl, NotSupportedError
from hidet.implement.search_space import ProductSpace, AtomSpace, SpaceChoice
from hidet.ir.builders import TaskBuilder, FunctionBuilder, StmtBuilder
from hidet.ir.dialects.compute import TensorInput, TensorCompute, compute, ReduceCompute, reduce_sum
from hidet.ir.dialects.lowlevel import Address
from hidet.ir.dialects.pattern import TaskPattern, any_const_int
from hidet.ir.expr import Call, TensorElement, var, tensor_var
from hidet.ir.func import IRModule, sync_threads
from hidet.ir.node import Node
from hidet.ir.stmt import LetStmt, EvaluateStmt, ForStmt
from hidet.ir.task import Task, ThreadBlock, Warp
from hidet.ir.type import scalar_type, TensorType, Scope, RegisterScope


class MatmulSetting:
    def __init__(self, block_layout, block_k, warp_layout, warp_k, regs_a_scope, regs_b_scope, regs_c_scope):
        self.block_layout: TaskLayout = block_layout
        self.block_k = block_k
        self.warp_layout: TaskLayout = warp_layout
        self.warp_k = warp_k
        self.regs_a_scope: RegisterScope = regs_a_scope
        self.regs_b_scope: RegisterScope = regs_b_scope
        self.regs_c_scope: RegisterScope = regs_c_scope


def default_setting():
    from hidet.implement.cuda.layout.concrete import WarpLayout4x8
    block_layout = row_major_layout(4, 2)
    warp_layout = (full_layout(2, 2) * WarpLayout4x8()) * full_layout(4, 4)
    regs_a_scope = RegisterScope(
        global2local=(lambda i, j: (i % 4 + (i // 16) * 4, 0)),
        local2global=(lambda tid, i, j: (i // 4 * 16 + tid // 16 * 8 + tid % 2 * 4 + i % 4, 0)),
        local_shape=(8, 1)
    )
    regs_b_scope = RegisterScope(
        global2local=(lambda i, j: (0, j % 4 + (j // 32) * 4)),
        local2global=(lambda tid, i, j: (0, j // 4 * 32 + tid % 16 // 2 * 4 + j % 4)),
        local_shape=(1, 8)
    )
    regs_c_scope = RegisterScope(
        global2local=(lambda i, j: (i // 16 * 4 + i % 4, j // 32 * 4 + j % 4)),
        local2global=(lambda tid, i, j: (i // 4 * 16 + tid // 16 * 8 + tid % 2 * 4 + i % 4,
                                         j // 4 * 32 + tid % 16 // 2 * 4 + j % 4)),
        local_shape=(8, 8)
    )
    return MatmulSetting(block_layout, 8, warp_layout, 1, regs_a_scope, regs_b_scope, regs_c_scope)


matmul_settings = [default_setting()]


def sync_threads_stmt():
    return EvaluateStmt(Call(sync_threads(), []))


@register_impl('cuda_block_static_matmul_nopipe_implementer')
class CudaBlockStaticMatmulNopipeImplementer(Implementer):
    """
    Without software pipeline:
        for block_tile_idx in range(task_k / block_k)
            gmem -> smem
            block_sync
            for frag_tile_idx in range(block_k / warp_k)
                smem -> regs
                regs -> regs (compute)
            block_sync
    """

    """
    TODO:
    With software pipeline:
        gmem -> current smem
        inc gmem
        sync
        current smem -> current regs
        inc smem
        sync
        for block_tile_idx in range(task_k / block_k) - 1
            current gmem -> next smem
            inc gmem
            for frag_tile_idx in range(block_k / warp_k)
                if is last warp tile:
                    swap smem
                current smem -> next regs
                inc smem
                current regs -> accumulate regs (compute)
                swap regs
        for frag_tile_idx in range(block_k / warp_k)
            if not last warp tile:
                current smem -> next regs
            inc smem
            current regs -> accumulate regs (compute)
            swap regs
    """

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
        self.A_strides = [any_const_int(), any_const_int()]
        self.B_strides = [any_const_int(), any_const_int()]
        self.C_strides = [any_const_int(), any_const_int()]
        self.A_type = TensorType(Scope('global'), scalar_type('float32'), shape=None, strides=self.A_strides)
        self.B_type = TensorType(Scope('global'), scalar_type('float32'), shape=None, strides=self.B_strides)
        self.C_type = TensorType(Scope('global'), scalar_type('float32'), shape=None, strides=self.C_strides)

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
            except (AssertionError, NotSupportedError, NotImplementedError):
                continue
            else:
                ir_module.include(sub_ir_module)
        if len(ir_module.functions) == 0:
            raise NotImplementedError('Can not find a setting to implement task {}'.format(task.name))
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

        assert block_n * warp_n == task_n
        assert block_m * warp_m == task_m
        assert task_k % (block_k * warp_k) == 0  # currently, we only implement perfect fit case

        # other constants
        assert block_size % 32 == 0
        num_warps = block_size // 32

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
        smem_A_type = TensorType(scope='shared', scalar_type=A.dtype, strides=[1, block_n])  # column major, TODO: add to search space
        smem_B_type = TensorType(scope='shared', scalar_type=B.dtype, strides=[block_m, 1])  # row major

        regs_A_type = TensorType(scope=setting.regs_a_scope, scalar_type=A_dtype)
        regs_B_type = TensorType(scope=setting.regs_b_scope, scalar_type=B_dtype)
        regs_C_type = TensorType(scope=setting.regs_c_scope, scalar_type=C_dtype)

        # define subtasks
        with TaskBuilder(f'{task.name}.a.g2s.block', ThreadBlock(block_size), ir_module) as a_g2s:
            gmem_frag_A = TensorInput('gmem_frag_A', A.dtype)
            gmem_frag_A_type = TensorType('global', A.dtype, strides=A_type.strides)
            a_g2s_cmpt = compute('smem_A', shape=[block_m * warp_m, block_k * warp_k], fcompute=lambda i, j: gmem_frag_A[i, j])
            a_g2s.set_computation(a_g2s_cmpt)
            a_g2s.append_param(gmem_frag_A, gmem_frag_A_type)
            a_g2s.append_param(a_g2s_cmpt, smem_A_type)
        print(a_g2s.name + " OK!")

        with TaskBuilder(f'{task.name}.b.g2s.block', ThreadBlock(block_size), ir_module) as b_g2s:
            gmem_frag_B = TensorInput('gmem_frag_B', B.dtype)
            gmem_frag_B_type = TensorType('global', B.dtype, strides=B_type.strides)
            b_g2s_cmpt = compute('smem_B', shape=[block_k * warp_k, block_n * warp_n], fcompute=lambda i, j: gmem_frag_B[i, j])
            b_g2s.set_computation(b_g2s_cmpt)
            b_g2s.append_param(gmem_frag_B, gmem_frag_B_type)
            b_g2s.append_param(b_g2s_cmpt, smem_B_type)

        with TaskBuilder(f'{task.name}.a.s2r.warp', Warp(), ir_module) as a_s2r:
            smem_frag_A = TensorInput('smem_frag_A', A.dtype)
            smem_frag_A_type = TensorType('shared', A.dtype, strides=smem_A_type.strides)
            a_s2r_cmpt = compute('regs_A', shape=[warp_m, warp_k], fcompute=lambda i, j: smem_frag_A[i, j])
            a_s2r.set_computation(a_s2r_cmpt)
            a_s2r.append_param(smem_frag_A, smem_frag_A_type)
            a_s2r.append_param(a_s2r_cmpt, regs_A_type)

        with TaskBuilder(f'{task.name}.b.s2r.warp', Warp(), ir_module) as b_s2r:
            smem_frag_B = TensorInput('smem_frag_B', B.dtype)
            smem_frag_B_type = TensorType('shared', B.dtype, strides=smem_B_type.strides)
            b_s2r_cmpt = compute('regs_B', shape=[warp_k, warp_n], fcompute=lambda i, j: smem_frag_B[i, j])
            b_s2r.set_computation(b_s2r_cmpt)
            b_s2r.append_param(smem_frag_B, smem_frag_B_type)
            b_s2r.append_param(b_s2r_cmpt, regs_B_type)

        with TaskBuilder(f'{task.name}.compute.warp', Warp(), ir_module) as ab2c:
            regs_A_input = TensorInput('regs_A', A.dtype)
            regs_B_input = TensorInput('regs_B', B.dtype)
            axis_k = var('k')
            fcompute = lambda i, j: reduce_sum(regs_A_input[i, axis_k] * regs_B_input[axis_k, j], axis=axis_k, shape=[warp_k])
            ab2c_cmpt = compute('regs_C', shape=[warp_m, warp_n], fcompute=fcompute)
            ab2c.set_computation(ab2c_cmpt)
            ab2c.append_param(regs_A_input, regs_A_type)
            ab2c.append_param(regs_B_input, regs_B_type)
            ab2c.append_param(ab2c_cmpt, regs_C_type)

        with TaskBuilder(f'{task.name}.r2g.warp', Warp(), ir_module) as c_r2g:
            regs_C_input = TensorInput('regs_C', C_dtype)
            c_r2g_cmpt = compute('gmem_C', shape=[warp_m, warp_n], fcompute=lambda i, j: regs_C_input[i, j])
            c_r2g.set_computation(c_r2g_cmpt)
            c_r2g.append_param(regs_C_input, regs_C_type)
            c_r2g.append_param(c_r2g_cmpt, C_type)

        # define function
        with FunctionBuilder(task.name) as fb:
            # declare params
            gmem_A = tensor_var('A', shape=[task_m, task_k], scope='global', dtype=A_dtype)
            gmem_B = tensor_var('B', shape=[task_k, task_n], scope='global', dtype=B_dtype)
            gmem_C = tensor_var('C', shape=[task_m, task_n], scope='global', dtype=C_dtype)
            fb.extend_params([gmem_A, gmem_B, gmem_C])

            # declare A, B shared memory
            smem_A = tensor_var('smem_A', shape=[block_n, block_k], scope='shared', dtype=A_dtype)
            smem_B = tensor_var('smem_B', shape=[block_n, block_k], scope='shared', dtype=B_dtype)
            fb.extend_local_vars([smem_A, smem_B])

            # declare A, B, C registers
            regs_A = tensor_var('regs_A', shape=setting.regs_a_scope.local_shape, scope=setting.regs_a_scope, dtype=A_dtype)
            regs_B = tensor_var('regs_B', shape=setting.regs_b_scope.local_shape, scope=setting.regs_b_scope, dtype=B_dtype)
            regs_C = tensor_var('regs_C', shape=setting.regs_c_scope.local_shape, scope=setting.regs_c_scope, dtype=C_dtype)
            fb.extend_local_vars([regs_A, regs_B, regs_C])

            # predefined variables
            thread_idx = var('threadIdx.x')

            # function body
            sb = StmtBuilder()
            warp_idx = var('warp_id')
            sb.append(LetStmt(warp_idx, thread_idx // 32))
            sb.enter_body()
            block_tile_idx = var('block_tile_idx')
            sb.append(ForStmt(loop_var=block_tile_idx, extent=task_k // (block_k * warp_k)))
            with sb.for_body():
                # gmem -> smem
                sb.append(a_g2s(Address(TensorElement(gmem_A, [0, block_tile_idx * (block_k * warp_k)])), smem_A))
                sb.append(b_g2s(Address(TensorElement(gmem_B, [block_tile_idx * (block_k * warp_k), 0])), smem_B))
                # sync
                sb.append(sync_threads_stmt())
                warp_tile_idx = var('warp_tile_idx')
                sb.append(ForStmt(loop_var=warp_tile_idx, extent=block_k))
                with sb.for_body():
                    # smem -> regs
                    sb.append(a_s2r(Address(TensorElement(smem_A, [warp_idx // 2 * 32, warp_tile_idx])), regs_A))
                    sb.append(a_s2r(Address(TensorElement(smem_B, [warp_tile_idx, warp_idx % 2 * 64])), regs_B))
                    # compute
                    sb.append(ab2c(regs_A, regs_B, regs_C))
                    # sync
                    sb.append(sync_threads_stmt())
            # regs -> gmem
            sb.append(c_r2g(regs_C, Address(TensorElement(gmem_C, [warp_idx // 2 * 32, warp_idx % 2 * 64]))))

            sb.exit_body()  # end of let warp_id
            fb.set_body(sb.finish())

            # attrs
            fb.extend_attrs({'worker': task.worker, 'label': 'matmul128x128x8'})

        func = fb.get()
        ir_module.add(func.name, func)






