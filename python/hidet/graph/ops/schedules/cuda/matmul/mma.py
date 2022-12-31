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
import contextlib
from typing import List, Tuple, Union, Optional, Sequence, TypeVar

import os

import hidet.cuda
from hidet import option
from hidet.ir.builders import FunctionBuilder, StmtBuilder
from hidet.ir.expr import Var, LogicalAnd, Equal, if_then_else, convert, Expr, tensor_var, cast, TensorSlice
from hidet.ir.expr import tensor_pointer_var
from hidet.ir.func import IRModule
from hidet.ir.primitives import syncthreads, thread_idx, block_idx
from hidet.ir.primitives.cuda.mma import MmaConfig, mma_sync
from hidet.ir.mapping import row_spatial, row_repeat
from hidet.ir.layout import row_layout, data_layout
from hidet.ir.stmt import BufferStoreStmt, IfStmt, Stmt, DeclareStmt, DeclareScope
from hidet.ir.type import DataType, data_type
from hidet.utils import prod
from hidet.graph.ops.definitions.matmul import BatchMatmulTask
from hidet.graph.ops.schedules.resolve import resolve_ir_modules
from hidet.graph.ops.schedules.common import params_from_task, Schedule, NotSupportedError
from hidet.graph.ops.schedules.cuda.common import get_transfer_task_map
from hidet.transforms.tools import fuse_and_pack

T = TypeVar('T', bound=Tuple)


def tuple_divide(lhs: T, rhs: T) -> T:
    assert len(lhs) == len(rhs) and all(a % b == 0 for a, b in zip(lhs, rhs))
    return tuple(a // b for a, b in zip(lhs, rhs))


class MatmulMmaSchedule(Schedule):
    def __init__(self, block_shape, warp_shape, mma_config: MmaConfig):
        self.block_shape: Tuple[int, int, int] = tuple(block_shape)
        self.warp_shape: Tuple[int, int, int] = tuple(warp_shape)
        self.mma_shape: Tuple[int, int, int] = (mma_config.m, mma_config.n, mma_config.k)
        self.mma_config: MmaConfig = mma_config

        self.check_divisible(self.block_shape, self.warp_shape)
        self.check_divisible(self.warp_shape, self.mma_shape)
        self.warp_count: Tuple[int, int, int] = tuple_divide(self.block_shape, self.warp_shape)
        self.mma_count: Tuple[int, int, int] = tuple_divide(self.warp_shape, self.mma_shape)
        self.threads: int = prod(self.warp_count) * 32
        self.check(self.threads <= 1024)

        self.block_m, self.block_n, self.block_k = self.block_shape
        self.warp_m, self.warp_n, self.warp_k = self.warp_shape
        self.mma_m, self.mma_n, self.mma_k = self.mma_config.m, self.mma_config.n, self.mma_config.k
        self.warp_count_m, self.warp_count_n, self.warp_count_k = self.warp_count
        self.mma_count_m, self.mma_count_n, self.mma_count_k = self.mma_count

        self.a_g2s_map, self.regs_a_ldg_layout = get_transfer_task_map(
            task_shape=[self.block_m, self.block_k], num_workers=self.threads, ranks=[0, 1]
        )
        self.b_g2s_map, self.regs_b_ldg_layout = get_transfer_task_map(
            task_shape=[self.block_k, self.block_n], num_workers=self.threads, ranks=[0, 1]
        )
        self.smem_a_layout = data_layout([2, self.block_m, self.block_k], ranks=[0, 1, 2])
        self.smem_b_layout = data_layout([2, self.block_k, self.block_n], ranks=[0, 1, 2])
        self.smem_c_layout = data_layout([self.block_m, self.block_n], ranks=[0, 1])
        self.regs_a_layout = row_layout(2, self.mma_count_m, mma_config.a_elements)
        self.regs_b_layout = row_layout(2, self.mma_count_n, mma_config.b_elements)
        self.regs_c_layout = row_layout(self.mma_count_m, self.mma_count_n, mma_config.c_elements)
        self.smem_storage_nbytes = max(
            (self.smem_a_layout.size + self.smem_b_layout.size) * data_type(mma_config.input_dtype).nbytes,
            self.smem_c_layout.size * data_type(mma_config.output_dtype).nbytes,
        )
        self.used_registers = (
            (
                self.regs_a_layout.size
                + self.regs_b_layout.size
                + self.regs_a_ldg_layout.size
                + self.regs_b_ldg_layout.size
            )
            * data_type(mma_config.input_dtype).nbytes
            + self.regs_c_layout.size * data_type(mma_config.output_dtype).nbytes
        ) // 4 + 24
        self.used_registers = (self.used_registers + 7) // 8 * 8
        self.check(self.smem_storage_nbytes <= 48 * 1024)
        self.check(self.used_registers <= 255)
        self.check(self.used_registers * self.threads <= hidet.cuda.properties().regsPerBlock)

    @staticmethod
    def resolve_mma_type(a_dtype: DataType, b_dtype: DataType, c_dtype: DataType):
        dtype_rank = {'float16': 0, 'bfloat16': 1, 'tfloat32': 2, 'float32': 4}
        ab_rank = max(dtype_rank[a_dtype.name], dtype_rank[b_dtype.name])
        if ab_rank <= dtype_rank['float16']:
            if c_dtype == 'float32':
                return 'mma_f16_f32'
            else:
                return 'mma_f16_f16'
        elif ab_rank <= dtype_rank['bfloat16']:
            return 'mma_bf16_f32'
        else:
            return 'mma_tf32_f32'

    @staticmethod
    def schedules(task: BatchMatmulTask, space_level: int):
        # ta, tb = task.attributes['ta'], task.attributes['tb']
        ta, tb = False, False
        if ta or tb:
            raise NotImplementedError()

        mma_type = task.attributes['mma']  # like 'wmma_f16_f32' or 'wmma'
        if mma_type == 'mma':
            a, b, c = task.inputs[0], task.inputs[1], task.outputs[0]
            a_dtype, b_dtype, c_dtype = [t.ttype.dtype for t in [a, b, c]]
            mma_type = MatmulMmaSchedule.resolve_mma_type(a_dtype, b_dtype, c_dtype)

        assert mma_type.startswith('mma')
        if space_level == 0:
            default_schedule = {
                # 'mma_f16_f16': MatmulMmaSchedule([128, 128, 16], [64, 64, 16], MmaConfig.m16n8k8_f16_f16()),
                'mma_f16_f16': MatmulMmaSchedule([64, 128, 16], [64, 64, 16], MmaConfig.m16n8k8_f16_f16()),
                'mma_f16_f32': MatmulMmaSchedule([128, 64, 16], [64, 64, 16], MmaConfig.m16n8k8_f16_f32()),
                'mma_bf16_f32': MatmulMmaSchedule([128, 64, 16], [64, 64, 16], MmaConfig.m16n8k8_bf16_f32()),
                'mma_tf32_f32': MatmulMmaSchedule([64, 64, 16], [32, 32, 16], MmaConfig.m16n8k8_tf32_f32()),
            }
            return [default_schedule[mma_type]]
        else:
            schedules: List[MatmulMmaSchedule] = []
            for mma_config in MmaConfig.all():
                head, input_dtype, output_dtype = mma_type.split('_')  # pylint: disable=unused-variable
                if mma_config.input_dtype != input_dtype or mma_config.output_dtype != output_dtype:
                    continue
                for block_m in [16, 32, 64, 128, 256] if space_level == 2 else [64, 128, 256]:
                    for block_n in [8, 16, 32, 64, 128] if space_level == 2 else [64, 128]:
                        for block_k in [8, 16, 32]:
                            for warp_m in [16, 32, 64] if space_level == 2 else [32, 64]:
                                for warp_n in [8, 16, 32, 64] if space_level == 2 else [32, 64]:
                                    for warp_k in [8, 16, 32]:
                                        with contextlib.suppress(NotSupportedError):
                                            schedules.append(
                                                MatmulMmaSchedule(
                                                    block_shape=[block_m, block_n, block_k],
                                                    warp_shape=[warp_m, warp_n, warp_k],
                                                    mma_config=mma_config,
                                                )
                                            )
                                            # print(len(schedules))
            if len(schedules) == 0:
                raise ValueError('Can not find a valid schedule for {}'.format(mma_type))
            return schedules

    def check_divisible(self, lhs: Sequence[int], rhs: Sequence[int]):
        if len(lhs) != len(rhs):
            raise NotSupportedError(self, 'length does not match, {} vs {}'.format(len(lhs), len(rhs)))
        for a, b in zip(lhs, rhs):
            if a % b != 0:
                raise NotSupportedError(self, 'not divisible, lhs {}, rhs {}'.format(lhs, rhs))

    def keys(self) -> List[Tuple[str, Union[int, float, str]]]:
        return [
            ('block', '{}x{}x{}'.format(self.block_m, self.block_n, self.block_k)),
            ('warp', '{}x{}x{}'.format(self.warp_m, self.warp_n, self.warp_k)),
            ('mma', str(self.mma_config)),
        ]

    def derived_keys(self) -> List[Tuple[str, Union[int, float, str]]]:
        return [
            ('warp_count', '{}x{}x{}'.format(self.warp_count_m, self.warp_count_n, self.warp_count_k)),
            ('mma_count', '{}x{}x{}'.format(self.mma_count_m, self.mma_count_n, self.mma_count_k)),
            ('threads', self.threads),
            ('smem_bytes', self.smem_storage_nbytes),
            ('registers', self.used_registers),
        ]


def batched_matmul_cuda_schedule_mma(task: BatchMatmulTask, working_dir: str) -> IRModule:
    all_schedules = MatmulMmaSchedule.schedules(task, space_level=option.get_option('search_space'))
    ir_modules = []
    for sch in all_schedules:
        ir_modules.append(batched_matmul_cuda_with_given_schedule(task, sch))

    return resolve_ir_modules(
        ir_modules=ir_modules,
        schedules=all_schedules,
        func_name=task.name,
        target_device='cuda',
        output_dir=os.path.join(working_dir, './resolve'),
        parallel=True,
        verbose=True,
    )


def batched_matmul_cuda_with_given_schedule(task: BatchMatmulTask, sch: MatmulMmaSchedule) -> IRModule:
    m_size, n_size, k_size = task.m_size, task.n_size, task.k_size
    m_tile_size, n_tile_size, k_tile_size = sch.block_shape  # pylint: disable=unused-variable
    m_tiles = (m_size + m_tile_size - 1) // m_tile_size
    n_tiles = (n_size + n_tile_size - 1) // n_tile_size
    grid_map = row_spatial(m_tiles, n_tiles)

    a_dtype = data_type(sch.mma_config.input_dtype)
    b_dtype = data_type(sch.mma_config.input_dtype)
    c_dtype = data_type(sch.mma_config.output_dtype)

    a_zero, b_zero, c_zero = [convert(0.0, dtype) for dtype in [a_dtype, b_dtype, c_dtype]]

    with FunctionBuilder(
        name=task.name + '_grid',
        kind='cuda_kernel',
        grid_dim=(grid_map.num_workers, task.batch_size),
        block_dim=sch.threads,
        label=str(sch),
    ) as fb:
        # declare params
        params = params_from_task(task)
        fb.extend_params(params)
        gmem_a, gmem_b, gmem_c = [param[block_idx('y'), :, :] for param in params]

        # declare local variables
        smem_a = tensor_pointer_var('smem_a', dtype=a_dtype, layout=sch.smem_a_layout)
        smem_b = tensor_pointer_var('smem_b', dtype=b_dtype, layout=sch.smem_b_layout)
        smem_c = tensor_pointer_var('smem_c', dtype=c_dtype, layout=sch.smem_c_layout)
        smem_storage = tensor_var('smem_storage', shape=[sch.smem_storage_nbytes], dtype='uint8')
        fb += DeclareStmt(smem_storage, scope=DeclareScope.Shared)
        fb += DeclareStmt(smem_a, init=cast(~smem_storage[0], ~a_dtype))
        fb += DeclareStmt(smem_b, init=cast(~smem_storage[smem_a.type.tensor_type.storage_bytes()], ~b_dtype))
        fb += DeclareStmt(smem_c, init=cast(~smem_storage[0], ~c_dtype))

        regs_a = tensor_var('regs_a', dtype=a_dtype, layout=sch.regs_a_layout)
        regs_b = tensor_var('regs_b', dtype=b_dtype, layout=sch.regs_b_layout)
        regs_c = tensor_var('regs_c', dtype=c_dtype, layout=sch.regs_c_layout)
        regs_ldg_a = tensor_var('regs_ldg_a', dtype=a_dtype, layout=sch.regs_a_ldg_layout)
        regs_ldg_b = tensor_var('regs_ldg_b', dtype=b_dtype, layout=sch.regs_b_ldg_layout)
        fb += DeclareStmt(regs_a)
        fb += DeclareStmt(regs_b)
        fb += DeclareStmt(regs_c)
        fb += DeclareStmt(regs_ldg_a)
        fb += DeclareStmt(regs_ldg_b)

        # initialize regs c
        for i, j, p in row_repeat(sch.mma_count_m, sch.mma_count_n, sch.mma_config.c_elements)[0]:
            fb += BufferStoreStmt(regs_c, (i, j, p), c_zero)

        with fb.lets(['bi', 'bj'], grid_map.single_task_of(block_idx())) as (bi, bj):
            block_k_tiles = (k_size + sch.block_k - 1) // sch.block_k
            block_offset_m, block_offset_n = [idx * extent for idx, extent in zip([bi, bj], [sch.block_m, sch.block_n])]
            # transfer first block
            fb += copy(
                gmem_a[block_offset_m:, :],
                regs_ldg_a,
                sch.a_g2s_map,
                src_pred=lambda i, k: LogicalAnd(block_offset_m + i < m_size, k < k_size),
                def_value=a_zero,
                cast_dtype=a_dtype,
            )
            fb += copy(regs_ldg_a, smem_a[0], sch.a_g2s_map)
            fb += copy(
                gmem_b[:, block_offset_n:],
                regs_ldg_b,
                sch.b_g2s_map,
                src_pred=lambda k, j: LogicalAnd(k < k_size, block_offset_n + j < n_size),
                def_value=b_zero,
                cast_dtype=b_dtype,
            )
            fb += copy(regs_ldg_b, smem_b[0], sch.b_g2s_map)
            fb += syncthreads()
            fb += load_regs_a(smem_a[0], regs_a[0], sch)
            fb += load_regs_b(smem_b[0], regs_b[0], sch)

            # main loop
            with fb.for_loop('k0', block_k_tiles) as k0:

                def body(fb, ko):
                    with fb.for_loop('k1', sch.mma_count_k) as k1:
                        with fb.if_then(Equal(k1, 0)):
                            block_offset_k = (k0 + 1) * sch.block_k
                            fb += copy(
                                gmem_a[block_offset_m:, block_offset_k:],
                                regs_ldg_a,
                                sch.a_g2s_map,
                                src_pred=lambda i, k: LogicalAnd(
                                    block_offset_m + i < m_size, block_offset_k + k < k_size
                                ),
                                def_value=a_zero,
                                cast_dtype=a_dtype,
                            )
                            fb += copy(
                                gmem_b[block_offset_k:, block_offset_n:],
                                regs_ldg_b,
                                sch.b_g2s_map,
                                src_pred=lambda k, j: LogicalAnd(
                                    block_offset_k + k < k_size, block_offset_n + j < n_size
                                ),
                                def_value=b_zero,
                                cast_dtype=b_dtype,
                            )
                        with fb.if_then(Equal(k1, sch.mma_count_k - 1)):
                            fb += copy(regs_ldg_a, smem_a[(k0 + 1) % 2], sch.a_g2s_map)
                            fb += copy(regs_ldg_b, smem_b[(k0 + 1) % 2], sch.b_g2s_map)
                            fb += syncthreads()
                            fb += load_regs_a(smem_a[(k0 + 1) % 2], regs_a[(k1 + ko + 1) % 2], sch)
                            fb += load_regs_b(smem_b[(k0 + 1) % 2], regs_b[(k1 + ko + 1) % 2], sch)
                        with fb.otherwise():
                            fb += load_regs_a(smem_a[k0 % 2, :, (k1 + 1) * sch.mma_k :], regs_a[(k1 + ko + 1) % 2], sch)
                            fb += load_regs_b(smem_b[k0 % 2, (k1 + 1) * sch.mma_k :, :], regs_b[(k1 + ko + 1) % 2], sch)
                        fb += warp_mma(regs_a[(k1 + ko) % 2], regs_b[(k1 + ko) % 2], regs_c, sch)

                if sch.mma_count_k % 2 == 0:
                    body(fb, 0)
                else:
                    with fb.if_then(Equal(k0 % 2, 0)):
                        body(fb, 0)
                    with fb.otherwise():
                        body(fb, 1)

            # write back
            fb += write_back(
                regs_c,
                smem_c,
                gmem_c[block_offset_m:, block_offset_n:],
                m_size - block_offset_m,
                n_size - block_offset_n,
                sch,
            )

    func = fb.func
    ir_module = IRModule(funcs={func.name: func}, task=task)
    return fuse_and_pack(ir_module, func, task)


def load_regs_a(smem_a: Union[Var, TensorSlice], regs_a: Var, sch: MatmulMmaSchedule) -> Stmt:
    # smem_a: Tensor[block_m, block_k]
    # regs_a: Tensor[mma_count_m, mma_a_elements]
    # will copy the [0, warp_k] in block_k range
    sb = StmtBuilder()
    warp_id, lane_id = thread_idx() / 32, thread_idx() % 32
    # pylint: disable=unused-variable
    for warp_i, warp_j, warp_k in row_spatial(sch.warp_count_m, sch.warp_count_n, sch.warp_count_k).on(warp_id):
        for mma_i in range(sch.mma_count_m):
            for p, (ii, kk) in enumerate(sch.mma_config.a_load_map.on(lane_id)):
                sb += BufferStoreStmt(
                    buf=regs_a,
                    indices=[mma_i, p],
                    value=smem_a[warp_i * sch.warp_m + mma_i * sch.mma_m + ii, warp_k * sch.warp_k + kk],
                )
    return sb.finish()


def load_regs_b(smem_b: Union[Var, TensorSlice], regs_b: Var, sch: MatmulMmaSchedule) -> Stmt:
    sb = StmtBuilder()
    warp_id, lane_id = thread_idx() / 32, thread_idx() % 32
    # pylint: disable=unused-variable
    for warp_i, warp_j, warp_k in row_spatial(sch.warp_count_m, sch.warp_count_n, sch.warp_count_k).on(warp_id):
        for mma_j in range(sch.mma_count_n):
            for p, (kk, jj) in enumerate(sch.mma_config.b_load_map.on(lane_id)):
                sb += BufferStoreStmt(
                    buf=regs_b,
                    indices=[mma_j, p],
                    value=smem_b[warp_k * sch.warp_k + kk, warp_j * sch.warp_n + mma_j * sch.mma_n + jj],
                )
    return sb.finish()


def warp_mma(regs_a, regs_b, regs_c, sch: MatmulMmaSchedule) -> Stmt:
    # regs_a: Tensor[mma_count_m, a_elements]
    # regs_b: Tensor[mma_count_n, b_elements]
    # regs_c: Tensor[mma_count_m, mma_count_n, c_elements]
    sb = StmtBuilder()
    for mma_i in range(sch.mma_count_m):
        for mma_j in range(sch.mma_count_n):
            sb += mma_sync(sch.mma_config, ~regs_a[mma_i, 0], ~regs_b[mma_j, 0], ~regs_c[mma_i, mma_j, 0])
    return sb.finish()


def write_back(regs_c, smem_c, gmem_c, bound_m, bound_n, sch: MatmulMmaSchedule) -> Stmt:
    # regs_c: Tensor[mma_count_m, mma_count_n, c_elements]
    # smem_c: Tensor[block_m, block_n]
    # gmem_c: Tensor[block_m, block_n]
    sb = StmtBuilder()
    warp_id, lane_id = thread_idx() / 32, thread_idx() % 32
    for warp_k_round in range(sch.warp_count_k):
        for warp_i, warp_j, warp_k in row_spatial(sch.warp_count_m, sch.warp_count_n, sch.warp_count_k).on(warp_id):
            with sb.if_then(Equal(warp_k, warp_k_round)):
                for mma_i, mma_j in row_repeat(sch.mma_count_m, sch.mma_count_n).on(0):
                    for p, (ii, jj) in enumerate(sch.mma_config.c_store_map.on(lane_id)):
                        i = warp_i * sch.warp_m + mma_i * sch.mma_m + ii
                        j = warp_j * sch.warp_n + mma_j * sch.mma_n + jj
                        with sb.if_then(LogicalAnd(i < bound_m, j < bound_n)):
                            if sch.warp_count_k == 1:
                                sb += BufferStoreStmt(gmem_c, [i, j], regs_c[mma_i, mma_j, p])
                            else:
                                if warp_k_round == 0:
                                    sb += BufferStoreStmt(smem_c, [i, j], regs_c[mma_i, mma_j, p])
                                elif 0 < warp_k_round < sch.warp_count_k - 1:
                                    sb += BufferStoreStmt(smem_c, [i, j], regs_c[mma_i, mma_j, p] + smem_c[i, j])
                                else:
                                    sb += BufferStoreStmt(gmem_c, [i, j], regs_c[mma_i, mma_j, p] + smem_c[i, j])
        if warp_k_round + 1 != sch.warp_count_k:
            sb += syncthreads()
    return sb.finish()


def copy(
    src,
    dst,
    mapping,
    src_pred=None,
    dst_pred=None,
    def_value: Optional[Union[Expr, float]] = 0.0,
    worker_idx=None,
    cast_dtype=None,
):
    if worker_idx is None:
        worker_idx = thread_idx()
    sb = StmtBuilder()
    for indices in mapping(worker_idx):
        value = src[indices]
        if cast_dtype is not None:
            value = cast(value, cast_dtype)
        if src_pred:
            value = if_then_else(src_pred(*indices), value, def_value)
        stmt = BufferStoreStmt(dst, indices, value)
        if dst_pred:
            stmt = IfStmt(dst_pred(*indices), stmt)
        sb += stmt
    return sb.finish()
