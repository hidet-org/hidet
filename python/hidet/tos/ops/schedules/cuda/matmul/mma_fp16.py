import os
from typing import List, Tuple, Union

import hidet
from hidet.ir.func import IRModule
from hidet.ir.primitives.cuda.mma import MmaConfig
from hidet.ir.task import TaskContext
from hidet.tos.ops.definitions.matmul.matmul import MatmulTask
from hidet.tos.ops.schedules.common import Schedule, NotSupportedError
from hidet.tos.ops.schedules.resolve import resolve_ir_modules


class MatmulMmaFp16Schedule(Schedule):
    def __init__(
            self,
            block_m: int,
            block_n: int,
            block_k: int,
            warp_m: int,
            warp_n: int,
            warp_k: int,
            mma_config: MmaConfig
    ):
        self.block_m: int = block_m
        self.block_n: int = block_n
        self.block_k: int = block_k
        self.warp_m: int = warp_m
        self.warp_n: int = warp_n
        self.warp_k: int = warp_k
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
    # load_smem_a_map = repeat(block_m // (threads // (block_k // 8)), 1).spatial(threads // (block_k // 8), block_k // 8)
    # load_smem_b_map = repeat(block_k // (threads // (block_n // 8)), 1).spatial(threads // (block_n // 8), block_n // 8)

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
                                    for mma_config in [
                                        MmaConfig.m16n8k16_f16_f16(),
                                        MmaConfig.m16n8k8_f16_f16()
                                    ]:
                                        try:
                                            sch.append(MatmulMmaFp16Schedule(
                                                block_m, block_n, block_k, warp_m, warp_n, warp_k, mma_config
                                            ))
                                        except NotSupportedError:
                                            pass
            return sch

    def keys(self) -> List[Tuple[str, Union[int, float, str]]]:
        return [
            ('block', '{}x{}x{}'.format(self.block_m, self.block_n, self.block_k)),
            ('warp', '{}x{}x{}'.format(self.warp_m, self.warp_n, self.warp_k)),
            ('mma', str(self.mma_config))
        ]

    def derived_keys(self) -> List[Tuple[str, Union[int, float, str]]]:
        return [
            ('warp_count', '{}x{}x{}'.format(self.warp_count_m, self.warp_count_n, self.warp_count_k)),
            ('mma_count', '{}x{}x{}'.format(self.mma_count_m, self.mma_count_n, self.mma_count_k)),
        ]


def batched_matmul_cuda_schedule_mma_fp16(task: MatmulTask) -> IRModule:
    ctx = TaskContext.current()
    default_resolve_out_dir = os.path.join('./outs/resolve', task.name, 'batched_matmul_mma_fp16_{}x{}x{}x{}'.format(task.batch_size, task.m_size, task.k_size, task.n_size))
    resolve_out_dir = ctx.resolve_out_dir if ctx.resolve_out_dir else default_resolve_out_dir

    all_schedules = MatmulMmaFp16Schedule.schedules(task, space_level=ctx.space_level)
    ir_modules = []
    for sch in all_schedules:
        ir_modules.append(gemm_mma_fp16_cp_async_ldmatrix_opt_kernel(task, sch))

    return resolve_ir_modules(
        ir_modules=ir_modules,
        schedules=all_schedules,
        output_dir=resolve_out_dir,
        parallel=True,
        verbose=True
    )


def gemm_mma_fp16_cp_async_ldmatrix_opt_kernel(
        task: MatmulTask,
        sch: MatmulMmaFp16Schedule
) -> IRModule:
    from hidet.lang import f16, spatial, repeat, tensor, attr, cast, col_spatial, view, u32, tensor_pointer
    from hidet.lang.layout import row_layout, DataLayout
    from hidet.lang.mapping import repeat, spatial
    from hidet.lang.cuda import blockIdx, threadIdx, syncthreads, dynamic_shared_memory
    from hidet.lang.cuda import mma_sync, cp_async, cp_async_wait_all, ldmatrix

    # optimize for 128x768x3072
    # mma_config = MmaConfig.m16n8k16_f16_f16()
    # block_m, block_n, block_k = 128, 128, 64
    # warp_m, warp_n, warp_k = 64, 32, 64
    bs, m_size, n_size, k_size = task.batch_size, task.m_size, task.n_size, task.k_size
    mma_config = sch.mma_config
    block_m, block_n, block_k = sch.block_m, sch.block_n, sch.block_k

    warp_m, warp_n, warp_k = sch.warp_m, sch.warp_n, sch.warp_k
    warp_count_m, warp_count_n, warp_count_k = block_m // warp_m, block_n // warp_n, block_k // warp_k
    mma_m, mma_n, mma_k = mma_config.m, mma_config.n, mma_config.k  # 16, 8, 16
    mma_count_m, mma_count_n, mma_count_k = warp_m // mma_m, warp_n // mma_n, warp_k // mma_k
    threads = warp_count_m * warp_count_n * warp_count_k * 32
    assert block_m % warp_m == block_n % warp_n == block_k % warp_k == 0
    assert warp_m % mma_m == warp_n % mma_n == warp_k % mma_k == 0
    smem_a_type = tensor(
        'shared', 'float16', shape=[block_m, block_k],
        layout=row_layout(block_m, block_k // 8).swizzle(1) * row_layout(1, 8)
    )
    smem_b_type = tensor(
        'shared', 'float16', shape=[block_k, block_n],
        layout=row_layout(block_k // 8, block_n // 64) * row_layout(8, 8).swizzle(1) * row_layout(1, 8)
    )
    load_smem_a_map = repeat(block_m // (threads // (block_k // 8)), 1).spatial(threads // (block_k // 8), block_k // 8)
    load_smem_b_map = repeat(block_k // (threads // (block_n // 8)), 1).spatial(threads // (block_n // 8), block_n // 8)

    with hidet.script_module(task) as module:
        @hidet.script
        def load_regs_a(
                k1: int,
                smem_a: smem_a_type,
                regs_a: tensor('register', 'float16', [mma_count_m, mma_config.a_elements])
        ):
            warp_id, lane_id = threadIdx.x / 32, threadIdx.x % 32
            for wi, wj, wk in spatial(warp_count_m, warp_count_n, warp_count_k).on(warp_id):
                for mi in range(mma_count_m):
                    p, q = col_spatial(16, 2).map(lane_id)
                    row_addr = ~smem_a[wi * warp_m + mi * mma_m + p, wk * warp_k + k1 * mma_k + q * 8]
                    b32_regs = view(~regs_a[mi, 0], u32[4])
                    ldmatrix(
                        regs=[b32_regs[0], b32_regs[1], b32_regs[2], b32_regs[3]],
                        smem_addr=row_addr,
                        shared_space_addr=False,
                        trans=False
                    )

        @hidet.script
        def load_regs_b(
                k1: int,
                smem_b: smem_b_type,
                regs_b: tensor('register', 'float16', [mma_count_n, mma_config.b_elements])
        ):
            warp_id, lane_id = threadIdx.x / 32, threadIdx.x % 32
            for wi, wj, wk in spatial(warp_count_m, warp_count_n, warp_count_k).on(warp_id):
                for mj in range(mma_count_n):
                    p, q = col_spatial(16, 2).map(lane_id)
                    # have not used q as we only use the address of the first 16 threads to load 2 of 8x8 f16 matrix.
                    row_addr = ~smem_b[wk * warp_k + k1 * mma_k + p, wj * warp_n + mj * mma_n]
                    regs = view(~regs_b[mj, 0], u32[2])
                    ldmatrix(
                        regs=[regs[0], regs[1]],
                        smem_addr=row_addr,
                        trans=True
                    )

        @hidet.script
        def warp_mma(
                regs_a: tensor('register', 'float16', [mma_count_m, mma_config.a_elements]),
                regs_b: tensor('register', 'float16', [mma_count_n, mma_config.b_elements]),
                regs_c: tensor('register', 'float16', [mma_count_m, mma_count_n, mma_config.c_elements])
        ):
            for mi, mj in repeat(mma_count_m, mma_count_n).on(0):
                mma_sync(mma_config, ~regs_a[mi, 0], ~regs_b[mj, 0], ~regs_c[mi, mj, 0])

        @hidet.script
        def store_c(
                regs_c: tensor('register', 'float16', [mma_count_m, mma_count_n, mma_config.c_elements]),
                c: tensor('global', 'float16', [bs, m_size, n_size])
        ):
            warp_id, lane_id = threadIdx.x / 32, threadIdx.x % 32
            offset_m, offset_n = blockIdx.x * block_m, blockIdx.y * block_n
            gmem_c = c[blockIdx.z, offset_m:, offset_n:]
            for k_round in range(warp_count_k):
                for wi, wj, wk in spatial(warp_count_m, warp_count_n, warp_count_k).on(warp_id):
                    if wk == k_round:
                        for mi, mj in repeat(mma_count_m, mma_count_n).on(0):
                            p = 0
                            for i, j in mma_config.c_store_map.on(lane_id):
                                gmem_c.write([wi * warp_m + mi * mma_m + i,
                                              wj * warp_n + mj * mma_n + j],
                                             regs_c[mi, mj, p],
                                             protected=True)
                                p += 1

        @hidet.script
        def load_smem_a(
                k0: int,
                a: f16[bs, m_size, k_size],
                smem_a: smem_a_type
        ):
            offset_m, offset_n, offset_k = blockIdx.x * block_m, blockIdx.y * block_n, k0 * block_k
            gmem_a = a[blockIdx.z, offset_m:, offset_k:]
            for i, k_seg in load_smem_a_map.on(threadIdx.x):
                k = k_seg * 8
                src_size = 0 if (offset_m + i >= m_size or offset_k + k >= k_size) else min(k_size - (offset_k + k), 8)
                cp_async(~smem_a[i, k], ~gmem_a[i, k], cp_size=16, src_size=src_size * 2, cache_level='global')

        @hidet.script
        def load_smem_b(
                k0: int,
                b: f16[bs, k_size, n_size],
                smem_b: smem_b_type
        ):
            offset_m, offset_n, offset_k = blockIdx.x * block_m, blockIdx.y * block_n, k0 * block_k
            gmem_b = b[blockIdx.z, offset_k:, offset_n:]
            for k, j_seg in load_smem_b_map.on(threadIdx.x):
                j = j_seg * 8
                src_size = 0 if (offset_k + k >= k_size or offset_n + j >= n_size) else min(n_size - (offset_n + j), 8)
                cp_async(~smem_b[k, j], ~gmem_b[k, j], cp_size=16, src_size=src_size * 2, cache_level='global')

        @hidet.script
        def matmul_grid(a: f16[bs, m_size, k_size], b: f16[bs, k_size, n_size], c: f16[bs, m_size, n_size]):
            # matrix multiplication, using mma instruction
            attr.cuda_grid_dim = (m_size + block_m - 1) // block_m, (n_size + block_n - 1) // block_n, bs
            attr.cuda_block_dim = threads
            attr.cuda_dynamic_smem_bytes = 2 * (block_m + block_n) * block_k * 2    # the second 2 means '2 bytes per float16'
            # smem_storage = dyn_smem_storage
            smem_a = tensor_pointer('shared', 'float16', shape=[2, block_m, block_k], layout=DataLayout.concat(row_layout(2), smem_a_type.layout))
            smem_b = tensor_pointer('shared', 'float16', shape=[2, block_k, block_n], layout=DataLayout.concat(row_layout(2), smem_b_type.layout))
            # smem_a = cast(~smem_storage[0], ~f16)
            # smem_b = cast(~smem_storage[2 * block_m * block_k * 2], ~f16)
            smem_a = dynamic_shared_memory(byte_offset=0, dtype=f16)
            smem_b = dynamic_shared_memory(byte_offset=2 * block_m * block_k * 2, dtype=f16)
            regs_a = tensor('register', 'float16', [mma_count_m, mma_config.a_elements])
            regs_b = tensor('register', 'float16', [mma_count_n, mma_config.b_elements])
            regs_c = tensor('register', 'float16', [mma_count_m, mma_count_n, mma_config.c_elements])

            for i, j, p in repeat(mma_count_m, mma_count_n, mma_config.c_elements).on(0):
                regs_c[i, j, p] = 0.0

            load_smem_a(0, a, ~smem_a[0, 0, 0])
            load_smem_b(0, b, ~smem_b[0, 0, 0])
            cp_async_wait_all()
            syncthreads()
            for k0 in range((k_size + block_k - 1) // block_k):
                load_smem_a(k0 + 1, a, ~smem_a[(k0 + 1) % 2, 0, 0])
                load_smem_b(k0 + 1, b, ~smem_b[(k0 + 1) % 2, 0, 0])
                for k1 in range(mma_count_k):
                    load_regs_a(k1, ~smem_a[k0 % 2, 0, 0], regs_a)
                    load_regs_b(k1, ~smem_b[k0 % 2, 0, 0], regs_b)
                    warp_mma(regs_a, regs_b, regs_c)
                cp_async_wait_all()
                syncthreads()
            store_c(regs_c, c)

    ir_module = module.ir_module()
    return ir_module
