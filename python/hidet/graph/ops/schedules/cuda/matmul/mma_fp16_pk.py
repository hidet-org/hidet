import os
from typing import List, Tuple, Union

import hidet
from hidet.ir.func import IRModule
from hidet.ir.primitives.cuda.mma import MmaConfig
from hidet.ir.task import TaskContext
from hidet.graph.ops.definitions.matmul.matmul import MatmulTask
from hidet.graph.ops.schedules.common import Schedule, NotSupportedError
from hidet.graph.ops.schedules.resolve import resolve_ir_modules
from hidet.transforms.tools import fuse_and_pack


class MatmulMmaFp16PkSchedule(Schedule):
    def __init__(
            self,
            block_m: int,
            block_n: int,
            block_k: int,
            warp_m: int,
            warp_n: int,
            warp_k: int,
            split_k_size: int,
            mma_config: MmaConfig
    ):
        self.block_m: int = block_m
        self.block_n: int = block_n
        self.block_k: int = block_k
        self.warp_m: int = warp_m
        self.warp_n: int = warp_n
        self.warp_k: int = warp_k
        self.split_k_size: int = split_k_size
        self.mma_config: MmaConfig = mma_config

        self.warp_count_m = block_m // warp_m
        self.warp_count_n = block_n // warp_n
        self.warp_count_k = block_k // warp_k

        self.mma_m = mma_config.m
        self.mma_n = mma_config.n
        self.mma_k = mma_config.k

        self.mma_count_m = warp_m // self.mma_m
        self.mma_count_n = warp_n // self.mma_n
        self.mma_count_k = warp_k // self.mma_k

        self.threads = self.warp_count_m * self.warp_count_n * self.warp_count_k * 32
        self.smem_nbytes = max(2 * (block_m + block_n) * block_k, block_m * block_n) * 2

        if not (block_m % warp_m == block_n % warp_n == block_k % warp_k == 0):
            raise NotSupportedError(self)
        if not (warp_m % mma_config.m == warp_n % mma_config.n == warp_k % mma_config.k == 0):
            raise NotSupportedError(self)
        if not (self.block_k % 8 == self.threads % (self.block_k // 8) == block_m % (self.threads // (self.block_k // 8)) == 0):
            raise NotSupportedError(self)
        if not (self.block_n % 8 == self.threads % (self.block_n // 8) == block_k % (self.threads // (self.block_n // 8)) == 0):
            raise NotSupportedError(self)
        if self.warp_count_k != 1:
            raise NotSupportedError(self)
        if mma_config is not MmaConfig.m16n8k16_f16_f16():
            raise NotSupportedError(self)
        from hidet.utils.cuda import max_smem_bytes_per_block
        if self.smem_nbytes > max_smem_bytes_per_block():
            raise NotSupportedError(self)

    @staticmethod
    def schedules(task, space_level=0):
        if space_level == 0:
            raise NotImplementedError()
        elif space_level == 1:
            raise NotImplementedError()
        else:
            sch = []
            for block_m in [64, 128]:
                for block_n in [64, 128]:
                    for block_k in [16, 32, 64]:
                        for warp_m in [32, 64]:
                            for warp_n in [32, 64]:
                                for warp_k in [16, 32, 64]:
                                    for split_k in [128, 256, 512, 1024]:
                                        for mma_config in [
                                            MmaConfig.m16n8k16_f16_f16(),
                                            MmaConfig.m16n8k8_f16_f16()
                                        ]:
                                            try:
                                                sch.append(MatmulMmaFp16PkSchedule(block_m, block_n, block_k, warp_m, warp_n, warp_k, split_k, mma_config))
                                            except NotSupportedError:
                                                pass
            return sch

    def keys(self) -> List[Tuple[str, Union[int, float, str]]]:
        return [
            ('block', '{}x{}x{}'.format(self.block_m, self.block_n, self.block_k)),
            ('warp', '{}x{}x{}'.format(self.warp_m, self.warp_n, self.warp_k)),
            ('split_k', '{}'.format(self.split_k_size)),
            ('mma', str(self.mma_config))
        ]

    def derived_keys(self) -> List[Tuple[str, Union[int, float, str]]]:
        return [
            ('warp_count', '{}x{}x{}'.format(self.warp_count_m, self.warp_count_n, self.warp_count_k)),
            ('mma_count', '{}x{}x{}'.format(self.mma_count_m, self.mma_count_n, self.mma_count_k)),
        ]


def batched_matmul_cuda_schedule_mma_fp16_pk(task: MatmulTask) -> IRModule:
    ctx = TaskContext.current()
    default_resolve_out_dir = os.path.join('./outs/resolve', task.name, 'batched_matmul_mma_fp16_{}x{}x{}x{}'.format(task.batch_size, task.m_size, task.k_size, task.n_size))
    resolve_out_dir = ctx.resolve_out_dir if ctx.resolve_out_dir else default_resolve_out_dir

    all_schedules = MatmulMmaFp16PkSchedule.schedules(task, space_level=ctx.space_level)
    ir_modules = []
    for sch in all_schedules:
        ir_modules.append(gemm_mma_fp16_cp_async_ldmatrix_opt_kernel(task, sch))

    return resolve_ir_modules(
        ir_modules=ir_modules,
        schedules=all_schedules,
        target_device='cuda',
        output_dir=resolve_out_dir,
        parallel=True,
        verbose=True
    )


def gemm_mma_fp16_cp_async_ldmatrix_opt_kernel(
        task: MatmulTask,
        sch: MatmulMmaFp16PkSchedule
) -> IRModule:
    from hidet.lang import f16, i32, spatial, repeat, tensor, attr, cast, col_spatial, view, u32, tensor_pointer, grid, var_of_function, static, void_p
    from hidet.lang.layout import row_layout, DataLayout
    from hidet.lang.mapping import repeat, spatial, auto_map
    from hidet.lang.cuda import blockIdx, threadIdx, syncthreads, dynamic_shared_memory, set_kernel_max_dynamic_smem_bytes
    from hidet.lang.cuda import mma_sync, cp_async, cp_async_wait_all, ldmatrix
    from hidet.lang.cuda import acquire_seq_semaphore, release_seq_semaphore

    batch_size, m_size, n_size, k_size = task.batch_size, task.m_size, task.n_size, task.k_size
    block_m, block_n, block_k = sch.block_m, sch.block_n, sch.block_k
    split_k_size = sch.split_k_size

    block_count_m, block_count_n, block_count_k = (m_size + block_m - 1) // block_m, (n_size + block_n - 1) // block_n, (k_size + split_k_size - 1) // split_k_size
    warp_m, warp_n, warp_k = sch.warp_m, sch.warp_n, sch.warp_k
    warp_count_m, warp_count_n, warp_count_k = sch.warp_count_m, sch.warp_count_n, sch.warp_count_k
    mma_config = sch.mma_config
    mma_m, mma_n, mma_k = mma_config.m, mma_config.n, mma_config.k
    mma_count_m, mma_count_n, mma_count_k = sch.mma_count_m, sch.mma_count_n, sch.mma_count_k
    threads = sch.threads
    smem_bytes = sch.smem_nbytes

    smem_a_type = tensor( 'shared', 'float16', shape=[block_m, block_k], layout=row_layout(block_m, block_k // 8).swizzle(1) * row_layout(1, 8))
    smem_b_type = tensor( 'shared', 'float16', shape=[block_k, block_n], layout=row_layout(block_k // 8, block_n // 64) * row_layout(8, 8).swizzle(1) * row_layout(1, 8))
    load_a_map = auto_map(block_m, block_k // 8, workers=threads)
    load_b_map = auto_map(block_k, block_n // 8, workers=threads)

    with hidet.script_module(task) as module:
        @hidet.script
        def load_regs_a(
                mi: int,
                k1: int,
                smem_a: smem_a_type,
                regs_a: tensor('register', 'float16', [mma_config.a_elements])
        ):
            warp_id, lane_id = threadIdx.x / 32, threadIdx.x % 32
            for wi, wj, wk in spatial(warp_count_m, warp_count_n, warp_count_k).on(warp_id):
                p, q = col_spatial(16, 2).map(lane_id)
                row_addr = ~smem_a[wi * warp_m + mi * mma_m + p, wk * warp_k + k1 * mma_k + q * 8]
                b32_regs = view(regs_a, u32[4])
                ldmatrix(
                    regs=[b32_regs[0], b32_regs[1], b32_regs[2], b32_regs[3]],
                    smem_addr=row_addr,
                    shared_space_addr=False,
                    trans=False
                )

        @hidet.script
        def load_regs_b(
                mj: int,
                k1: int,
                smem_b: smem_b_type,
                regs_b: tensor('register', 'float16', [mma_config.b_elements])
        ):
            warp_id, lane_id = threadIdx.x / 32, threadIdx.x % 32
            for wi, wj, wk in spatial(warp_count_m, warp_count_n, warp_count_k).on(warp_id):
                p, q = col_spatial(16, 2).map(lane_id)
                # have not used q as we only use the address of the first 16 threads to load 2 of 8x8 f16 matrix.
                row_addr = ~smem_b[wk * warp_k + k1 * mma_k + p, wj * warp_n + mj * mma_n]
                regs = view(regs_b, u32[2])
                ldmatrix(
                    regs=[regs[0], regs[1]],
                    smem_addr=row_addr,
                    trans=True
                )

        @hidet.script
        def warp_mma(
                regs_a: tensor('register', 'float16', [mma_config.a_elements]),
                regs_b: tensor('register', 'float16', [mma_config.b_elements]),
                regs_c: tensor('register', 'float16', [mma_config.c_elements])
        ):
            mma_sync(mma_config, regs_a, regs_b, regs_c)

        @hidet.script
        def store_c(
                regs_c: tensor('register', 'float16', [mma_count_m, mma_count_n, mma_config.c_elements]),
                c: tensor('global', 'float16', [batch_size, m_size, n_size]),
                locks: tensor('global', 'int32', [batch_size, block_count_m, block_count_n])
        ):
            split_k_tile, batch = spatial(block_count_k, batch_size).single_task_of(blockIdx.z)
            smem_c: tensor_pointer('shared', 'float16', [block_m, block_n])
            smem_c = dynamic_shared_memory(byte_offset=0, dtype='float16')

            warp_id, lane_id = threadIdx.x / 32, threadIdx.x % 32
            offset_m, offset_n = blockIdx.x * block_m, blockIdx.y * block_n
            lock = ~locks[batch, blockIdx.x, blockIdx.y]

            acquire_seq_semaphore(lock, split_k_tile)
            if split_k_tile == 0:
                for i, j in auto_map(block_m, block_n, workers=threads).on(threadIdx.x):
                    smem_c[i, j] = 0.0
            else:
                for i, j in auto_map(block_m, block_n, workers=threads).on(threadIdx.x):
                    smem_c[i, j] = c.read([batch, offset_m + i, offset_n + j], protected=True)
            syncthreads()
            for warp_k_round in range(warp_count_k):
                for wi, wj, wk in spatial(warp_count_m, warp_count_n, warp_count_k).on(warp_id):
                    if wk == warp_k_round:
                        for mi, mj in grid(mma_count_m, mma_count_n):
                            p = 0
                            for ii, jj in mma_config.c_store_map.on(lane_id):
                                smem_c[wi * warp_m + mi * mma_m + ii, wj * warp_n + mj * mma_n + jj] += regs_c[mi, mj, p]
                                p += 1
                syncthreads()
            for i, j in auto_map(block_m, block_n, workers=threads).on(threadIdx.x):
                c.write([batch, offset_m + i, offset_n + j], smem_c[i, j], protected=True)
            release_seq_semaphore(lock, (split_k_tile + 1) % block_count_k)

        @hidet.script
        def load_smem_a(
                k0: int,
                a: f16[batch_size, m_size, k_size],
                smem_a: smem_a_type
        ):
            split_k_tile, batch = spatial(block_count_k, batch_size).single_task_of(blockIdx.z)
            offset_m, offset_n, offset_k = blockIdx.x * block_m, blockIdx.y * block_n, k0 * block_k + split_k_tile * split_k_size
            gmem_a = a[batch, offset_m:, offset_k:]
            for i, k_seg in load_a_map.on(threadIdx.x):
                k = k_seg * 8
                src_size = 0 if (offset_m + i >= m_size or offset_k + k >= k_size) else min(k_size - (offset_k + k), 8)
                cp_async(~smem_a[i, k], ~gmem_a[i, k], cp_size=16, src_size=src_size * 2, cache_level='global')

        @hidet.script
        def load_smem_b(
                k0: int,
                b: f16[batch_size, k_size, n_size],
                smem_b: smem_b_type
        ):
            split_k_tile, batch = spatial(block_count_k, batch_size).single_task_of(blockIdx.z)
            offset_m, offset_n, offset_k = blockIdx.x * block_m, blockIdx.y * block_n, k0 * block_k + split_k_tile * split_k_size
            gmem_b = b[batch, offset_k:, offset_n:]
            for k, j_seg in load_b_map.on(threadIdx.x):
                j = j_seg * 8
                src_size = 0 if (offset_k + k >= k_size or offset_n + j >= n_size) else min(n_size - (offset_n + j), 8)
                cp_async(~smem_b[k, j], ~gmem_b[k, j], cp_size=16, src_size=src_size * 2, cache_level='global')

        @hidet.script
        def matmul_grid(
                a: f16[batch_size, m_size, k_size],
                b: f16[batch_size, k_size, n_size],
                c: f16[batch_size, m_size, n_size],
                locks: i32[batch_size, block_count_m, block_count_n]
        ):
            # matrix multiplication, using mma instruction
            attr.cuda_grid_dim = (m_size + block_m - 1) // block_m, (n_size + block_n - 1) // block_n, batch_size * block_count_k
            attr.cuda_block_dim = threads
            attr.cuda_dynamic_smem_bytes = smem_bytes
            smem_a: tensor_pointer('shared', 'float16', shape=[2, block_m, block_k], layout=DataLayout.concat(row_layout(2), smem_a_type.layout))
            smem_b: tensor_pointer('shared', 'float16', shape=[2, block_k, block_n], layout=DataLayout.concat(row_layout(2), smem_b_type.layout))
            smem_a = dynamic_shared_memory(byte_offset=0, dtype=f16)
            smem_b = dynamic_shared_memory(byte_offset=2 * block_m * block_k * 2, dtype=f16)
            regs_a = tensor('register', 'float16', [2, mma_count_m, mma_config.a_elements])
            regs_b = tensor('register', 'float16', [2, mma_count_n, mma_config.b_elements])
            regs_c = tensor('register', 'float16', [mma_count_m, mma_count_n, mma_config.c_elements])

            for i, j, p in grid(mma_count_m, mma_count_n, mma_config.c_elements):
                regs_c[i, j, p] = 0.0

            load_smem_a(0, a, ~smem_a[0, 0, 0])
            load_smem_b(0, b, ~smem_b[0, 0, 0])
            cp_async_wait_all()
            syncthreads()
            for k0 in range((split_k_size + block_k - 1) // block_k):
                load_smem_a(k0 + 1, a, ~smem_a[(k0 + 1) % 2, 0, 0])
                load_smem_b(k0 + 1, b, ~smem_b[(k0 + 1) % 2, 0, 0])
                for mi in range(mma_count_m):
                    load_regs_a(mi, 0, ~smem_a[k0 % 2, 0, 0], ~regs_a[0, mi, 0])
                for mj in range(mma_count_n):
                    load_regs_b(mj, 0, ~smem_b[k0 % 2, 0, 0], ~regs_b[0, mj, 0])
                for mk in range(mma_count_k):
                    if mk + 1 < mma_count_k:
                        for mi in range(mma_count_m):
                            load_regs_a(mi, mk + 1, ~smem_a[k0 % 2, 0, 0], ~regs_a[(mk + 1) % 2, mi, 0])
                        for mj in range(mma_count_n):
                            load_regs_b(mj, mk + 1, ~smem_b[k0 % 2, 0, 0], ~regs_b[(mk + 1) % 2, mj, 0])
                    for mi, mj in grid(mma_count_m, mma_count_n):
                        warp_mma(~regs_a[mk % 2, mi, 0], ~regs_b[mk % 2, mj, 0], ~regs_c[mi, mj, 0])
                cp_async_wait_all()
                syncthreads()
            store_c(regs_c, c, locks)

        from hidet.lang.runtime import request_workspace

        @hidet.script
        def matmul(
                num_args: int,
                arg_types: ~i32,
                args: ~void_p
        ):
            attr.func_kind = 'packed_func'
            attr.packed_func = var_of_function(matmul_grid)
            locks = static(tensor_pointer('global', dtype='int32', shape=[batch_size, block_count_m, block_count_n]))
            if locks == 0:
                locks = request_workspace(nbytes=4 * batch_size * block_count_m * block_count_n, require_clean=True)
            if sch.smem_nbytes > 48 * 1024:
                set_kernel_max_dynamic_smem_bytes(var_of_function(matmul_grid), max_dynamic_smem_bytes=sch.smem_nbytes)
            assert num_args == 3
            assert arg_types[0] == 3
            assert arg_types[1] == 3
            assert arg_types[2] == 3
            matmul_grid(
                cast(args[0], ~f16),
                cast(args[1], ~f16),
                cast(args[2], ~f16),
                locks
            )
    return fuse_and_pack(module.ir_module(), matmul_grid, task)
