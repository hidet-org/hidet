from hidet.runtime import CompiledFunction
import hidet
from ..common import BenchResult, benchmark_run


def gemm_mma_fp16(bs: int, m: int, n: int, k: int, warmup, number, repeat) -> BenchResult:
    a = hidet.randint(2, shape=[bs, m, k], dtype='float16')
    b = hidet.randint(2, shape=[bs, k, n], dtype='float16')
    c = hidet.empty(shape=[bs, m, n], dtype='float16')
    func = gemm_mma_fp16_kernel(bs, m, n, k)
    results = benchmark_run(run_func=lambda: func(a, b, c), warmup=warmup, number=number, repeat=repeat)
    return BenchResult(
        latencies=results,
        outputs=[c],
        configs='manual'
    )


def gemm_mma_fp16_kernel(bs, m_size, n_size, k_size) -> CompiledFunction:
    from hidet.lang import f16, spatial, repeat, tensor
    from hidet.lang.layout import row_layout, col_layout, local_layout
    from hidet.lang.mapping import repeat, spatial
    from hidet.lang.cuda import MmaConfig, blockIdx, threadIdx, syncthreads

    # optimize for 128x768x3072
    mma_config = MmaConfig.m16n8k16_f16_f16()
    block_m, block_n, block_k = 128, 128, 16

    @hidet.script
    def gemm_mma_grid(a: f16[bs, m_size, k_size], b: f16[bs, k_size, n_size], c: f16[bs, m_size, n_size]):
        # matrix multiplication, using mma instruction
        offset_m, offset_n = blockIdx.x * block_m, blockIdx.y * block_n
        warp_id, lane_id = threadIdx.x // 32, threadIdx.x % 32
        smem_a = tensor('shared', 'float16', [block_m, block_k])
        smem_b = tensor('shared', 'float16', [block_k, block_n])
        regs_a = tensor('register', 'float16', [4, mma_config.a_elements])
        regs_b = tensor('register', 'float16', [8, mma_config.b_elements])
        regs_c = tensor('register', 'float16', [4, 8, mma_config.c_elements])

        for i, j, k, p in repeat(4, 8, 2, mma_config.c_elements):
            regs_c[i, j, k, p] = 0.0

        for k0 in range((k_size + block_k - 1) // block_k):
            offset_k = k0 * block_k
            gmem_a = a[blockIdx.z, offset_m:, offset_k:]
            gmem_b = b[blockIdx.z, offset_k:, offset_n:]
            for i, k in repeat(16, 1).spatial(8, 16).on(threadIdx.x):
                smem_a[i, k] = gmem_a.read([i, k], protected=True)
            for k, j in repeat(16, 1).spatial(1, 128).on(threadIdx.x):
                smem_b[k, j] = gmem_b.read([k, j], protected=True)
            syncthreads()
            p = 0
            for i, k in mma_config.a_load_map.on(lane_id):
                pass






