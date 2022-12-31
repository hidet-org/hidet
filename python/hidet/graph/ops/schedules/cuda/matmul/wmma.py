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
import os
from typing import List, Tuple, Union, Optional

import hidet.cuda
from hidet import option
from hidet.ir.builders import FunctionBuilder, StmtBuilder
from hidet.ir.expr import Var, LogicalAnd, Cast, if_then_else, convert, Expr, cast
from hidet.ir.func import IRModule
from hidet.ir.functors import simplify_to_int
from hidet.ir.mapping import TaskMapping, row_spatial, row_repeat
from hidet.ir.layout import DataLayout, row_layout, local_layout, data_layout
from hidet.ir.primitives import syncthreads, thread_idx, block_idx
from hidet.ir.primitives.cuda.wmma import WmmaConfig, wmma_load_a, wmma_load_b, wmma_mma, wmma_store, wmma_configs
from hidet.ir.stmt import BufferStoreStmt, IfStmt, DeclareStmt, DeclareScope
from hidet.ir.task import Task
from hidet.ir.type import data_type, tensor_type, PointerType, tensor_pointer_type
from hidet.graph.ops.definitions.matmul import BatchMatmulTask
from hidet.graph.ops.schedules.common import params_from_task, Schedule, NotSupportedError
from hidet.graph.ops.schedules.cuda.common import get_task_map, get_transfer_task_map
from hidet.graph.ops.schedules.resolve import resolve_ir_modules
from hidet.utils import prod
from hidet.transforms.tools import fuse_and_pack


def shape_prod(a_shape: List[int], b_shape: List[int]) -> List[int]:
    assert len(a_shape) == len(b_shape)
    return [a * b for a, b in zip(a_shape, b_shape)]


short_dtype_bytes = {'f16': 2, 'bf16': 2, 'tf32': 4, 'f32': 4}


def find_compatible_wmma_config(config: WmmaConfig):
    for wmma_config in wmma_configs:
        for a, b in zip(config, wmma_config):
            if a is None:
                continue
            if a != b:
                break
        else:
            return wmma_config
    raise ValueError('Can not find compatible wmma config for: {}'.format(config))


class MatmulSchedule(Schedule):
    def __init__(self, wmma_config: WmmaConfig, block_multiple=(4, 2), warp_multiple=(2, 2, 1)):
        self.block_multiple = block_multiple
        self.warp_multiple = warp_multiple
        self.wmma_config = wmma_config

        wmma_shape = wmma_config.shape[:2]
        warp_shape = shape_prod(warp_multiple[:2], wmma_shape)
        block_shape = shape_prod(block_multiple, warp_shape)
        wmma_k = wmma_config.shape[2]
        warp_k = warp_multiple[2] * wmma_k
        self.wmma_k = wmma_k
        self.warp_k = warp_k
        self.wmma_shape = wmma_shape
        self.warp_shape = warp_shape
        self.block_shape = block_shape

        # task layouts
        warp_size = 32
        warps = prod(block_multiple)
        threads = warps * warp_size
        self.check(threads <= 1024)
        self.threads = threads
        self.warp_map = row_spatial(*block_multiple) * row_repeat(*warp_multiple[:2])
        self.c_init_map = self.warp_map

        self.check(wmma_config.a_layout == 'row')
        self.a_g2s_map, self.regs_a_ldg_layout = get_transfer_task_map(
            task_shape=[block_shape[0], warp_k], num_workers=threads, ranks=[0, 1]
        )
        self.smem_a_layout = data_layout([2, block_shape[0], warp_k], ranks=[0, 1, 2])
        self.load_a_stride = warp_k

        self.check(wmma_config.b_layout == 'row')
        self.b_g2s_map, self.regs_b_ldg_layout = get_transfer_task_map(
            task_shape=[warp_k, block_shape[1]], num_workers=threads, ranks=[0, 1]
        )
        self.smem_b_layout = data_layout([2, warp_k, block_shape[1]], ranks=[0, 1, 2])
        self.load_b_stride = block_shape[1]

        self.c_s2g_warp_map = row_spatial(*wmma_shape)
        self.smem_c_layout = row_layout(*block_multiple) * local_layout(*warp_multiple[:2]) * row_layout(*wmma_shape)
        self.check(wmma_config.c_layout == 'row')
        self.store_c_stride = wmma_shape[1]

        # data layouts
        # TODO: The following layout would cause duplicated data loading from shared memory to register
        c_seg_layout = local_layout(*block_multiple) * row_layout(*warp_multiple[:2])
        ab_seg_layout = local_layout(*block_multiple) * local_layout(*warp_multiple[:2])
        self.regs_a_layout = DataLayout.concat(ab_seg_layout, row_layout(wmma_config.a_regs))
        self.regs_b_layout = DataLayout.concat(ab_seg_layout, row_layout(wmma_config.b_regs))
        self.regs_c_layout = DataLayout.concat(c_seg_layout, row_layout(wmma_config.c_regs))

        # analyze regs & smem usage
        reserved_regs = 48  # number of reserved registers for intermediate results
        self.used_num_regs_per_thread = (
            self.regs_a_layout.size
            + self.regs_b_layout.size
            + self.regs_c_layout.size
            + self.regs_a_ldg_layout.size
            + self.regs_b_ldg_layout.size
            + reserved_regs
        )
        self.used_num_regs_per_thread = (self.used_num_regs_per_thread + 7) // 8 * 8
        self.used_smem_bytes_per_block = max(
            (self.smem_a_layout.size + self.smem_b_layout.size) * short_dtype_bytes[wmma_config.a_dtype],
            self.smem_c_layout.size * short_dtype_bytes[wmma_config.c_dtype],
        )
        self.used_smem_bytes_per_block = (self.used_smem_bytes_per_block + 127) // 128 * 128

        self.check(self.used_num_regs_per_thread <= 255, 'registers per thread exceeded')
        self.check(
            self.used_num_regs_per_thread * threads <= hidet.cuda.properties().regsPerBlock,
            'registers in block exceeded',
        )
        self.check(
            self.used_smem_bytes_per_block <= hidet.cuda.properties().sharedMemPerBlock, 'shared memory exceeded'
        )

        self.use_dynamic_smem = self.used_smem_bytes_per_block > 48 * 1024
        resident_blocks = min(
            hidet.cuda.properties().sharedMemPerBlock // (self.used_num_regs_per_thread * threads),
            hidet.cuda.properties().sharedMemPerMultiprocessor // self.used_smem_bytes_per_block,
        )
        self.min_thread_blocks = resident_blocks
        self.resident_blocks = resident_blocks

    def keys(self) -> List[Tuple[str, Union[int, float, str]]]:
        return [
            ('block_multiple', '{}x{}'.format(*self.block_multiple)),
            ('warp_multiple', '{}x{}x{}'.format(*self.warp_multiple)),
            ('wmma_shape', 'm{}n{}k{}'.format(*self.wmma_config.shape)),
            ('wmma_type', '{}_{}'.format(self.wmma_config.a_dtype, self.wmma_config.c_dtype)),
        ]

    def derived_keys(self) -> List[Tuple[str, Union[int, float, str]]]:
        return [
            ('block', '{}x{}'.format(*self.block_shape)),
            ('threads', self.threads),
            ('regs', self.used_num_regs_per_thread),
            ('smem', self.used_smem_bytes_per_block),
            ('resident_blocks', self.resident_blocks),
        ]

    def check(self, cond, msg: str = ""):
        if not cond:
            raise NotSupportedError(self, msg)

    @staticmethod
    def schedules(task: Task, space_level: int = 0):
        wmma_type = task.attributes['mma']  # like 'wmma_f16_f32' or 'wmma'

        # choose a specific wmma type when needed
        if wmma_type == 'wmma':
            dtype_rank = {'float16': 0, 'bfloat16': 1, 'tfloat32': 2, 'float32': 4}
            a_dtype = task.inputs[0].ttype.dtype.name
            b_dtype = task.inputs[1].ttype.dtype.name
            c_dtype = task.outputs[0].ttype.dtype.name
            ab_rank = max(dtype_rank[a_dtype], dtype_rank[b_dtype])
            if ab_rank <= dtype_rank['float16']:
                if c_dtype == 'float32':
                    wmma_type = 'wmma_f16_f32'
                else:
                    wmma_type = 'wmma_f16_f16'
            elif ab_rank <= dtype_rank['bfloat16']:
                wmma_type = 'wmma_bf16_f32'
            else:
                wmma_type = 'wmma_tf32_f32'

        # generate schedule space according to space level
        if space_level in [0, 1]:
            default_shape = {
                'wmma_f16_f16': (16, 16, 16),
                'wmma_f16_f32': (16, 16, 16),
                'wmma_bf16_f32': (16, 16, 16),
                'wmma_tf32_f32': (16, 16, 8),
            }
            a_layout = 'row'
            b_layout = 'row'
            _, ab_dtype, c_dtype = wmma_type.split('_')
            wmma_config = find_compatible_wmma_config(
                WmmaConfig(
                    shape=default_shape[wmma_type],
                    a_dtype=ab_dtype,
                    b_dtype=ab_dtype,
                    c_dtype=c_dtype,
                    a_layout=a_layout,
                    b_layout=b_layout,
                    c_layout='row',
                    a_regs=None,
                    b_regs=None,
                    c_regs=None,
                )
            )
            return [MatmulSchedule(wmma_config, block_multiple=(2, 2), warp_multiple=(2, 2, 1))]
        elif space_level == 2:
            ret = []
            _, ab_dtype, c_dtype = wmma_type.split('_')
            for wmma_config in wmma_configs:
                if wmma_config.a_dtype != ab_dtype or wmma_config.b_dtype != ab_dtype or wmma_config.c_dtype != c_dtype:
                    continue
                for block_warps_x in [4, 2, 1]:
                    for block_warps_y in [4, 2, 1]:
                        block_warps = (block_warps_x, block_warps_y)
                        for warp_outer_x in [4, 2, 1]:
                            for warp_outer_y in [4, 2, 1]:
                                for warp_outer_k in [1, 2]:
                                    warp_outer = (warp_outer_x, warp_outer_y, warp_outer_k)
                                    try:
                                        ret.append(MatmulSchedule(wmma_config, block_warps, warp_outer))
                                    except NotSupportedError:
                                        pass
            if len(ret) == 0:
                raise ValueError('Can not find schedule for task: \n{}'.format(task))
            return ret
        else:
            raise ValueError('Space level {} must in [0, 1, 2].'.format(space_level))


def batched_matmul_cuda_schedule_wmma(task: BatchMatmulTask, working_dir: str) -> IRModule:
    all_schedules = MatmulSchedule.schedules(task, space_level=option.get_option('search_space'))
    ir_modules = []
    for schedule in all_schedules:
        ir_modules.append(batched_matmul_cuda_with_given_schedule(task, schedule))
    return resolve_ir_modules(
        ir_modules=ir_modules,
        schedules=all_schedules,
        func_name=task.name,
        target_device='cuda',
        output_dir=os.path.join(working_dir, './resolve'),
        parallel=True,
        verbose=True,
    )


def batched_matmul_cuda_with_given_schedule(task: BatchMatmulTask, schedule: MatmulSchedule) -> IRModule:
    ir_module = IRModule(task=task)
    sch = schedule

    dtype_short2long = {'f16': 'float16', 'bf16': 'bfloat16', 'tf32': 'tfloat32', 'f32': 'float32'}

    a_dtype = data_type(dtype_short2long[sch.wmma_config.a_dtype])
    b_dtype = data_type(dtype_short2long[sch.wmma_config.b_dtype])
    c_dtype = data_type(dtype_short2long[sch.wmma_config.c_dtype])

    batch_size = task.batch_size
    m_size, k_size, n_size = task.m_size, task.k_size, task.n_size

    warp_size = 32
    m_tile_size, n_tile_size = sch.block_shape
    m_tiles = (m_size + m_tile_size - 1) // m_tile_size
    n_tiles = (n_size + n_tile_size - 1) // n_tile_size
    grid_blocks_layout: TaskMapping = TaskMapping.row_major([m_tiles, n_tiles])

    # define function
    with FunctionBuilder(
        name=task.name + '_grid',
        kind='cuda_kernel',
        grid_dim=(grid_blocks_layout.num_workers, batch_size),
        block_dim=sch.threads,
        dynamic_smem_bytes=sch.used_smem_bytes_per_block if sch.use_dynamic_smem else 0,
        min_blocks=sch.min_thread_blocks,
        label=str(sch),
    ) as fb:
        # declare params
        params = params_from_task(task)
        gmem_a, gmem_b, gmem_c = params
        fb.extend_params(params)

        # declare local variables
        smem_a = Var('smem_a', tensor_pointer_type(a_dtype, layout=sch.smem_a_layout))
        smem_b = Var('smem_b', tensor_pointer_type(b_dtype, layout=sch.smem_b_layout))
        smem_c = Var('smem_c', tensor_pointer_type(c_dtype, layout=sch.smem_c_layout))
        if sch.use_dynamic_smem:
            # 'extern __shared__ uint8_t smem_storage[];' in c code
            smem_storage = Var(
                'smem_storage',
                PointerType(base_type=data_type('uint8'), specifiers=['extern', '__shared__'], use_bracket=True),
            )
            fb += DeclareStmt(smem_storage)
        else:
            smem_storage = Var('smem_storage', tensor_type(dtype='uint8', shape=[sch.used_smem_bytes_per_block]))
            fb += DeclareStmt(smem_storage, scope=DeclareScope.Shared)
        smem_a_bytes = simplify_to_int(smem_a.type.tensor_type.storage_bytes())
        fb += DeclareStmt(smem_a, Cast(~smem_storage[0], PointerType(a_dtype)))
        fb += DeclareStmt(smem_b, Cast(~smem_storage[smem_a_bytes], PointerType(b_dtype)))
        fb += DeclareStmt(smem_c, Cast(~smem_storage[0], PointerType(c_dtype)))

        # declare a, b, c registers
        reg_dtype = data_type('uint32')
        regs_a = Var('regs_a', tensor_type(reg_dtype, layout=schedule.regs_a_layout))
        regs_b = Var('regs_b', tensor_type(reg_dtype, layout=schedule.regs_b_layout))
        regs_c = Var('regs_c', tensor_type(reg_dtype, layout=schedule.regs_c_layout))
        regs_a_ldg = Var('regs_a_ldg', tensor_type(dtype=a_dtype, layout=schedule.regs_a_ldg_layout))
        regs_b_ldg = Var('regs_b_ldg', tensor_type(dtype=b_dtype, layout=schedule.regs_b_ldg_layout))
        fb += DeclareStmt(regs_a)
        fb += DeclareStmt(regs_b)
        fb += DeclareStmt(regs_c)
        fb += DeclareStmt(regs_a_ldg)
        fb += DeclareStmt(regs_b_ldg)

        a_default_value = convert(0.0, a_dtype)
        b_default_value = convert(0.0, b_dtype)
        acc_default_value = convert(0, regs_c.type.dtype)

        wmma = sch.wmma_config
        wmma_m, wmma_n, wmma_k = wmma.shape

        with fb.lets(['bi', 'bj'], grid_blocks_layout(block_idx())[0]) as (bi, bj):
            block_k_tiles = (k_size + sch.warp_k - 1) // sch.warp_k
            first_k_tile = k_size - (block_k_tiles - 1) * sch.warp_k
            block_offset = [idx * dim for idx, dim in zip([bi, bj], sch.block_shape)]
            # transfer first tile
            fb += copy(
                gmem_a[block_idx('y'), block_offset[0] :, :],
                regs_a_ldg,
                sch.a_g2s_map,
                src_predicate=lambda i, k: LogicalAnd.join(block_offset[0] + i < m_size, k < first_k_tile),
                default_value=a_default_value,
                cast_dtype=a_dtype,
            )
            fb += copy(regs_a_ldg, smem_a[0], layout=sch.a_g2s_map)
            fb += copy(
                gmem_b[block_idx('y'), :, block_offset[1] :],
                regs_b_ldg,
                sch.b_g2s_map,
                src_predicate=lambda k, j: LogicalAnd.join(k < first_k_tile, block_offset[1] + j < n_size),
                default_value=b_default_value,
                cast_dtype=b_dtype,
            )
            fb += copy(regs_b_ldg, smem_b[0], layout=sch.b_g2s_map)
            fb += syncthreads()
            # init regs c
            fb += init(regs_c, acc_default_value, sch)
            # main loop
            warp_idx = thread_idx() / 32
            with fb.for_loop('k0', block_k_tiles - 1) as k0:
                block_offset_k = k0 * sch.warp_k + first_k_tile
                fb += copy(
                    gmem_a[block_idx('y'), block_offset[0] :, block_offset_k:],
                    regs_a_ldg,
                    schedule.a_g2s_map,
                    src_predicate=lambda i, _: block_offset[0] + i < m_size,
                    default_value=a_default_value,
                    cast_dtype=a_dtype,
                )
                fb += copy(
                    gmem_b[block_idx('y'), block_offset_k:, block_offset[1] :],
                    regs_b_ldg,
                    schedule.b_g2s_map,
                    src_predicate=lambda _, j: block_offset[1] + j < n_size,
                    default_value=b_default_value,
                    cast_dtype=b_dtype,
                )
                for warp_offset_x, warp_offset_y in sch.warp_map(warp_idx):
                    regs_a_addr = ~regs_a[warp_offset_x, warp_offset_y, 0]
                    regs_b_addr = ~regs_b[warp_offset_x, warp_offset_y, 0]
                    regs_c_addr = ~regs_c[warp_offset_x, warp_offset_y, 0]
                    for k1 in range(sch.warp_multiple[2]):
                        fb += wmma_load_a(
                            sch.wmma_config,
                            regs_a_addr,
                            ~smem_a[k0 % 2, warp_offset_x * wmma_m, k1 * wmma_k],
                            sch.load_a_stride,
                        )
                        fb += wmma_load_b(
                            sch.wmma_config,
                            regs_b_addr,
                            ~smem_b[k0 % 2, k1 * wmma_k, warp_offset_y * wmma_n],
                            sch.load_b_stride,
                        )
                        fb += wmma_mma(sch.wmma_config, regs_a_addr, regs_b_addr, regs_c_addr)
                fb += copy(regs_a_ldg, smem_a[(k0 + 1) % 2], schedule.a_g2s_map)
                fb += copy(regs_b_ldg, smem_b[(k0 + 1) % 2], schedule.b_g2s_map)
                fb += syncthreads()
            with fb.let('block_k_tile', block_k_tiles - 1) as k0:
                for warp_offset_x, warp_offset_y in sch.warp_map(warp_idx):
                    regs_a_addr = ~regs_a[warp_offset_x, warp_offset_y, 0]
                    regs_b_addr = ~regs_b[warp_offset_x, warp_offset_y, 0]
                    regs_c_addr = ~regs_c[warp_offset_x, warp_offset_y, 0]
                    for k1 in range(sch.warp_multiple[2]):
                        fb += wmma_load_a(
                            sch.wmma_config,
                            regs_a_addr,
                            ~smem_a[k0 % 2, warp_offset_x * wmma_m, k1 * wmma_k],
                            sch.load_a_stride,
                        )
                        fb += wmma_load_b(
                            sch.wmma_config,
                            regs_b_addr,
                            ~smem_b[k0 % 2, k1 * wmma_k, warp_offset_y * wmma_n],
                            sch.load_b_stride,
                        )
                        fb += wmma_mma(sch.wmma_config, regs_a_addr, regs_b_addr, regs_c_addr)
            fb += syncthreads()  # we need this because smem_c shares the memory with smem_a and smem_b
            for warp_offset_x, warp_offset_y in sch.warp_map(warp_idx):
                offset_x = warp_offset_x * wmma_m
                offset_y = warp_offset_y * wmma_n
                smem_c_addr = ~smem_c[offset_x, offset_y]
                regs_c_addr = ~regs_c[warp_offset_x, warp_offset_y, 0]
                fb += wmma_store(sch.wmma_config, smem_c_addr, regs_c_addr, sch.store_c_stride)
                fb += copy(
                    src=smem_c[offset_x:, offset_y:],
                    dst=gmem_c[block_idx('y'), block_offset[0] + offset_x :, block_offset[1] + offset_y :],
                    layout=get_task_map(task_shape=(wmma_m, wmma_n), num_workers=32, ranks=[0, 1]),
                    dst_predicate=lambda i, j: LogicalAnd(
                        block_offset[0] + offset_x + i < m_size, block_offset[1] + offset_y + j < n_size
                    ),
                    worker_idx=thread_idx() % warp_size,
                )

    func = fb.get()
    ir_module = IRModule(funcs={func.name: func}, task=task)
    return fuse_and_pack(ir_module, func, task)


def init(dst, init_value, sch):
    sb = StmtBuilder()
    warp_idx = thread_idx() / 32
    for indices in sch.warp_map(warp_idx):
        for i in range(sch.wmma_config.c_regs):
            sb += BufferStoreStmt(dst, indices + (i,), init_value)
    return sb.finish()


def copy(
    src,
    dst,
    layout,
    src_predicate=None,
    dst_predicate=None,
    default_value: Optional[Union[Expr, float]] = 0.0,
    worker_idx=None,
    cast_dtype=None,
):
    if worker_idx is None:
        worker_idx = thread_idx()
    sb = StmtBuilder()
    for indices in layout(worker_idx):
        value = src[indices]
        if cast_dtype is not None:
            value = cast(value, cast_dtype)
        if src_predicate:
            value = if_then_else(src_predicate(*indices), value, default_value)
        stmt = BufferStoreStmt(dst, indices, value)
        if dst_predicate:
            stmt = IfStmt(dst_predicate(*indices), stmt)
        sb += stmt
    return sb.finish()
