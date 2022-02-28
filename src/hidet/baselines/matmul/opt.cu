// Acknowledgement: https://github.com/Yinghan-Li/YHs_Sample/blob/master/cuda/gemm/sgemm.cu

#include <cassert>
#include <cstdint>

#include <hidet/runtime.h>
#include <hidet/packedfunc.h>


__device__ __forceinline__
uint32_t smem_u32addr(const void *smem_ptr) {
    uint32_t addr;
    asm ("{.reg .u64 u64addr;\n"
         " cvta.to.shared.u64 u64addr, %1;\n"
         " cvt.u32.u64 %0, u64addr;}\n"
    : "=r"(addr)
    : "l"(smem_ptr)
    );

    return addr;
}

__device__ __forceinline__
void ldg32_nc(float &reg, const void *ptr, bool guard) {
    asm volatile (
    "{.reg .pred p;\n"
    " setp.ne.b32 p, %2, 0;\n"
    #if __CUDACC_VER_MAJOR__ >= 11 && __CUDACC_VER_MINOR__ >= 4 && \
    __CUDA_ARCH__ >= 750
    " @p ld.global.nc.L2::128B.f32 %0, [%1];}\n"
#else
    " @p ld.global.nc.f32 %0, [%1];}\n"
#endif
    : "=f"(reg)
    : "l"(ptr), "r"((int)guard)
    );
}

__device__ __forceinline__
void ldg32_nc_0(float &reg, const void *ptr, bool guard) {
    asm volatile (
    "{.reg .pred p;\n"
    " setp.ne.b32 p, %2, 0;\n"
    " @!p mov.b32 %0, 0;\n"
    #if __CUDACC_VER_MAJOR__ >= 11 && __CUDACC_VER_MINOR__ >= 4 && \
    __CUDA_ARCH__ >= 750
    " @p ld.global.nc.L2::128B.f32 %0, [%1];}\n"
#else
    " @p ld.global.nc.f32 %0, [%1];}\n"
#endif
    : "=f"(reg)
    : "l"(ptr), "r"((int)guard)
    );
}

__device__ __forceinline__
void stg32(const float &reg, const void *ptr, bool guard) {
    asm volatile (
    "{.reg .pred p;\n"
    " setp.ne.b32 p, %2, 0;\n"
    " @p st.global.f32 [%0], %1;}\n"
    : : "l"(ptr), "f"(reg), "r"((int)guard)
    );
}

__device__ __forceinline__
void lds128(float &reg0, float &reg1,
            float &reg2, float &reg3,
            const uint32_t &addr) {
    asm volatile (
    "ld.shared.v4.f32 {%0, %1, %2, %3}, [%4];\n"
    : "=f"(reg0), "=f"(reg1), "=f"(reg2), "=f"(reg3)
    : "r"(addr)
    );
}

__device__ __forceinline__
void sts32(const float &reg, const uint32_t &addr) {
    asm volatile (
    "st.shared.f32 [%0], %1;\n"
    : : "r"(addr), "f"(reg)
    );
}

__device__ __forceinline__
void sts128(const float &reg0, const float &reg1,
            const float &reg2, const float &reg3,
            const uint32_t &addr) {
    asm volatile (
    "st.shared.v4.f32 [%0], {%1, %2, %3, %4};\n"
    : : "r"(addr), "f"(reg0), "f"(reg1), "f"(reg2), "f"(reg3)
    );
}

struct StgFrag {
    float data[4][4];

    __device__ __forceinline__
    StgFrag(const float (&C_frag)[8][8], int tile_x, int tile_y) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                data[i][j] = C_frag[tile_y * 4 + i][tile_x * 4 + j];
            }
        }
    }
};

__device__ __noinline__
void C_tile_wb(StgFrag C_frag,
               float *C_stg_ptr,
               const float *C_lds_ptr,
               uint32_t C_sts_addr,
               uint32_t m,
               uint32_t n,
               uint32_t m_idx,
               uint32_t n_idx) {
    __syncthreads();

#pragma unroll
    for (int i = 0; i < 4; ++i) {
        sts128(C_frag.data[i][0],
               C_frag.data[i][1],
               C_frag.data[i][2],
               C_frag.data[i][3],
               C_sts_addr + i * 8 * sizeof(float4));
    }

    __syncthreads();

    uint32_t m_guard = m < m_idx ? 0 : m - m_idx;

#pragma unroll
    for (int i = 0; i < 16; ++i) {
        stg32(C_lds_ptr[i * 32],
              C_stg_ptr + i * n,
              i < m_guard && n_idx < n);
    }
}

/*
 * matrix A, B and C: row-major
 *
 * mma block:
 * thread block tile: m128n128k8
 * warp tile: m32n64k8
 * thread tile: m8n8k8
 * thread fragment:
 *     matrixA: 8x1 FP32
 *     matrixB: 1x8 FP32
 *
 * ----------------------------------------------------------------
 * thread block tile map:
 *
 *                                128
 *                    --|---------------------|
 *             B_tile  8|                     |
 *                    --|---------------------|
 *
 *  A_tile   | 8 |      |    64    |
 *         --|---|    --|----------|----------|
 *           |   |    32|  warp_0  |  warp_1  |
 *           |   |    --|----------|----------|
 *           |   |      |  warp_2  |  warp_3  |
 *        128|   |      |----------|----------|
 *           |   |      |  warp_4  |  warp_5  |
 *           |   |      |----------|----------|
 *           |   |      |  warp_6  |  warp_7  |
 *         --|---|      |----------|----------|
 *
 * ----------------------------------------------------------------
 * warp tile map:
 *

a a+1 a+2 a+3 a+H a+H+1 a+H+2 a+H+3 a+2H+1 ... a+7H a+7H+1 a+7H+2 a+7H+3
------------- --------------------- ...        -------------------------
thread 8g     thread 8g+1          ...         thread 8g+7
where g is the group index in the warp and H is the number of shared memory rows.
H % 32 = 4


 * 'z' thread map to avoid LDS.128 shared memory broadcast limitation.
 *
 *              |              32               ||
 *     B_frag --|---|---|---|---|---|---|---|---||---|---|---|---|---|---|---|---|
 *             1|///|   |   |   |   |   |   |   ||///|   |   |   |   |   |   |   |
 *            --|---|---|---|---|---|---|---|---||---|---|---|---|---|---|---|---|
 * A_frag       | 4 |                           ||
 *    | 1 |                                     ||
 *  --|---|--   |---|---|---|---|---|---|---|---||---|---------------------------|
 *    |///|4    |t0 |t2 |t4 |t6 |t8 |t10|t12|t14||t0 |                           |
 *    |---|--   |---|---|---|---|---|---|---|---||---|                           |
 *    |   |     |t1 |t3 |t5 |t7 |t9 |t11|t13|t15||                               |
 *  16|---|     |---|---|---|---|---|---|---|---||                               |
 *    |   |     |t16|t18|t20|t22|t24|t26|t28|t30||                               |
 *    |---|     |---|---|---|---|---|---|---|---||                               |
 *    |   |     |t17|t19|t21|t23|t25|t27|t29|t31||                               |
 *  ==|===|=====|===|===|===|===|===|===|===|===||===|============================
 *    |///|     |t0 |                           ||t0 |                           |
 *    |---|     |---|                           ||---|                           |
 *    |   |     |                               ||                               |
 *    |---|     |                               ||                               |
 *    |   |     |                               ||                               |
 *    |---|     |                               ||                               |
 *    |   |     |                               ||                               |
 *    |---|     |-------------------------------||-------------------------------|
 *
 */
static __global__ __launch_bounds__(256, 2)
void sgemm_128x128x8_kernel(const float *A,
                            const float *B,
                            float *C,
                            uint32_t m,
                            uint32_t n,
                            uint32_t k,
                            uint32_t A_ldg_step,    // k * sizeof(float)
                            uint32_t B_ldg_step) {  // n * sizeof(float) * 8
    /*
     * matrix A & B thread block tile shared memory (double buffer)
     * matrix A: 132 * 8 * 4Byte/item * double buffer = 4.125KB * 2
     * matrix B: 128 * 8 * 4Byte/item * double buffer = 8KB
     *
     * for double buffer faster switch, A_smem requires 8KB * 2 shared memory
     * and 16KB aligned, B_smem should be 8KB aligned, then the double buffer
     * can be switched by only 1 xor instruction:
     *     (uint32_t &)A_smem ^= 0x2000;
     *     (uint32_t &)B_smem ^= 0x1000;
     */
    __shared__ __align__(16 * 1024) char smem[16 * 1024];
    float *A_smem = reinterpret_cast<float *>(smem);
    float *B_smem = reinterpret_cast<float *>(smem + 8 * 1024);

    // A, B and C register fragment
    float A_frag[2][8];
    float B_frag[2][8];
    float C_frag[8][8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            C_frag[i][j] = 0;
        }
    }

    const uint32_t lane_id = threadIdx.x % 32;
    const uint32_t warp_id = threadIdx.x / 32;

    // 4x8 threads each warp for FFMA
    const uint32_t mma_tid_x = (lane_id / 2) % 8;
    const uint32_t mma_tid_y = (lane_id / 16) * 2 + (lane_id % 2);

    // A_tile & B_tile ldg pointer
    // A[blockIdx.y * 128 + threadIdx.x / 8 * 4][threadIdx.x % 8]
    const char *A_ldg_ptr = (const char *)(A + (blockIdx.y * 128 + threadIdx.x / 8 * 4) * k + threadIdx.x % 8);
    // B[threadIdx.x / 32][blockIdx.x * 128 + threadIdx.x % 32]
    const char *B_ldg_ptr = (const char *)(B + (threadIdx.x / 32) * n + blockIdx.x * 128 + threadIdx.x % 32);

    // A_tile & B_tile sts/lds pointer
    // using uint32_t pointer for faster double buffer switch
    // A_smem[threadIdx.x/8 * 4][threadIdx.x % 8]
    uint32_t A_sts_addr = smem_u32addr(A_smem + (threadIdx.x % 8) * 128 + (threadIdx.x / 8) * 4);
    // B_smem[threadIdx.x / 32][threadIdx.x % 32]
    uint32_t B_sts_addr = smem_u32addr(B_smem + (threadIdx.x / 32) * 128 + (threadIdx.x % 32));

    uint32_t A_lds_addr = smem_u32addr(A_smem + (warp_id / 2) * 32 + mma_tid_y * 4);
    uint32_t B_lds_addr = smem_u32addr(B_smem + (warp_id % 2) * 64 + mma_tid_x * 4);

    // ldg_guard to avoid LDG out of bound
    uint32_t A_ldg_guard = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        int m_idx = blockIdx.y * 128 + threadIdx.x / 8 * 4 + i;
        if (m_idx < m) {
            A_ldg_guard |= (1u << i);
        }
    }

    uint32_t B_ldg_guard = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        int n_idx = blockIdx.x * 128 + threadIdx.x % 32 + i * 32;
        if (n_idx < n) {
            B_ldg_guard |= (1u << i);
        }
    }

    float A_ldg_reg[4];
    float B_ldg_reg[4];

    // 1'st A&B tile loaded before the k_tile loop
    uint32_t k_tiles = (k + 7) / 8 - 1;

    // load 1'st tile to shared memory
    {
        uint32_t first_k_tile = k - k_tiles * 8;

#pragma unroll
        for (int i = 0; i < 4; ++i) {
            bool guard = (A_ldg_guard & (1u << i)) != 0 &&
                         threadIdx.x % 8 < first_k_tile;
            ldg32_nc_0(A_ldg_reg[i],
                       A_ldg_ptr + i * A_ldg_step,  // A[blockIdx.y * ][0]
                       guard);
        }

        sts128(A_ldg_reg[0], A_ldg_reg[1], A_ldg_reg[2], A_ldg_reg[3],
               A_sts_addr);

#pragma unroll
        for (int i = 0; i < 4; ++i) {
            bool guard = (B_ldg_guard & (1u << i)) != 0 &&
                         threadIdx.x / 32 < first_k_tile;
            ldg32_nc_0(B_ldg_reg[i],
                       B_ldg_ptr + i * 32 * sizeof(float),
                       guard);
        }

#pragma unroll
        for (int i = 0; i < 4; ++i) {
            sts32(B_ldg_reg[i], B_sts_addr + i * 32 * sizeof(float));
        }

        __syncthreads();

        // switch double buffer
        A_sts_addr ^= 0x1000;
        B_sts_addr ^= 0x1000;

        // ldg pointer for next tile
        A_ldg_ptr += first_k_tile * sizeof(float);
        B_ldg_ptr += n * first_k_tile * sizeof(float);
    }

    // load 1'st fragment
    lds128(A_frag[0][0], A_frag[0][1], A_frag[0][2], A_frag[0][3], A_lds_addr);
    lds128(A_frag[0][4], A_frag[0][5], A_frag[0][6], A_frag[0][7], A_lds_addr + 16 * sizeof(float));
    lds128(B_frag[0][0], B_frag[0][1], B_frag[0][2], B_frag[0][3], B_lds_addr);
    lds128(B_frag[0][4], B_frag[0][5], B_frag[0][6], B_frag[0][7], B_lds_addr + 32 * sizeof(float));

    // k_tiles loop
    for (; k_tiles > 0; --k_tiles) {
#pragma unroll
        for (int k_frag = 0; k_frag < 8; ++k_frag) {
            // store next A&B tile to shared memory
            if (k_frag == 7) {
                sts128(A_ldg_reg[0], A_ldg_reg[1], A_ldg_reg[2], A_ldg_reg[3],
                       A_sts_addr);
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    sts32(B_ldg_reg[i], B_sts_addr + i * 32 * sizeof(float));
                }

                __syncthreads();

                // switch double buffer
                A_lds_addr ^= 0x1000;
                B_lds_addr ^= 0x1000;
                A_sts_addr ^= 0x1000;
                B_sts_addr ^= 0x1000;

                // ldg pointer for next tile
                A_ldg_ptr += 8 * sizeof(float);
                B_ldg_ptr += B_ldg_step;
            }

            // load next A&B fragment from shared memory to register
            lds128(A_frag[(k_frag + 1) % 2][0],
                   A_frag[(k_frag + 1) % 2][1],
                   A_frag[(k_frag + 1) % 2][2],
                   A_frag[(k_frag + 1) % 2][3],
                   A_lds_addr + (k_frag + 1) % 8 * 128 * sizeof(float));
            lds128(A_frag[(k_frag + 1) % 2][4],
                   A_frag[(k_frag + 1) % 2][5],
                   A_frag[(k_frag + 1) % 2][6],
                   A_frag[(k_frag + 1) % 2][7],
                   A_lds_addr + ((k_frag + 1) % 8 * 128 + 16) * sizeof(float));
            lds128(B_frag[(k_frag + 1) % 2][0],
                   B_frag[(k_frag + 1) % 2][1],
                   B_frag[(k_frag + 1) % 2][2],
                   B_frag[(k_frag + 1) % 2][3],
                   B_lds_addr + (k_frag + 1) % 8 * 128 * sizeof(float));
            lds128(B_frag[(k_frag + 1) % 2][4],
                   B_frag[(k_frag + 1) % 2][5],
                   B_frag[(k_frag + 1) % 2][6],
                   B_frag[(k_frag + 1) % 2][7],
                   B_lds_addr + ((k_frag + 1) % 8 * 128 + 32) * sizeof(float));

            // load next A&B tile
            if (k_frag == 0) {
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    ldg32_nc(A_ldg_reg[i],
                             A_ldg_ptr + i * A_ldg_step,
                             (A_ldg_guard & (1u << i)) != 0);
                }

#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    ldg32_nc(B_ldg_reg[i],
                             B_ldg_ptr + i * 32 * sizeof(float),
                             (B_ldg_guard & (1u << i)) != 0);
                }
            }

            // FFMA loop
#pragma unroll
            for (int i = 0; i < 8; ++i) {
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    C_frag[i][j] += A_frag[k_frag % 2][i] *
                                    B_frag[k_frag % 2][j];
                }
            }
        }
    }

    // FFMA for the last tile
#pragma unroll
    for (int k_frag = 0; k_frag < 8; ++k_frag) {
        if (k_frag < 7) {
            // load next A&B fragment from shared memory to register
            lds128(A_frag[(k_frag + 1) % 2][0],
                   A_frag[(k_frag + 1) % 2][1],
                   A_frag[(k_frag + 1) % 2][2],
                   A_frag[(k_frag + 1) % 2][3],
                   A_lds_addr + (k_frag + 1) % 8 * 128 * sizeof(float));
            lds128(A_frag[(k_frag + 1) % 2][4],
                   A_frag[(k_frag + 1) % 2][5],
                   A_frag[(k_frag + 1) % 2][6],
                   A_frag[(k_frag + 1) % 2][7],
                   A_lds_addr + ((k_frag + 1) % 8 * 128 + 16) * sizeof(float));
            lds128(B_frag[(k_frag + 1) % 2][0],
                   B_frag[(k_frag + 1) % 2][1],
                   B_frag[(k_frag + 1) % 2][2],
                   B_frag[(k_frag + 1) % 2][3],
                   B_lds_addr + (k_frag + 1) % 8 * 128 * sizeof(float));
            lds128(B_frag[(k_frag + 1) % 2][4],
                   B_frag[(k_frag + 1) % 2][5],
                   B_frag[(k_frag + 1) % 2][6],
                   B_frag[(k_frag + 1) % 2][7],
                   B_lds_addr + ((k_frag + 1) % 8 * 128 + 32) * sizeof(float));
        }

        // FFMA loop
#pragma unroll
        for (int i = 0; i < 8; ++i) {
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                C_frag[i][j] += A_frag[k_frag % 2][i] *
                                B_frag[k_frag % 2][j];
            }
        }
    }

    // C_tile write back, reuse A&B tile shared memory buffer
    uint32_t C_sts_addr = smem_u32addr((float4 *)(smem + warp_id * 2048) +
                                       mma_tid_y * 4 * 8 + mma_tid_x);
    const float *C_lds_ptr = (float *)(smem + warp_id * 2048) + lane_id;

    uint32_t m_idx = blockIdx.y * 128 + warp_id / 2 * 32;
    uint32_t n_idx = blockIdx.x * 128 + warp_id % 2 * 64 + lane_id;

    float *C_stg_ptr = C + m_idx * n + n_idx;

    if (m_idx >= m) {
        return;
    } else if (m_idx + 32 <= m) {
        uint32_t n_guard = n < n_idx ? 0 : n - n_idx;

#pragma unroll
        for (int i = 0; i < 2; ++i) {
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                __syncthreads();

#pragma unroll
                for (int p = 0; p < 4; ++p) {
                    sts128(C_frag[i * 4 + p][j * 4],
                           C_frag[i * 4 + p][j * 4 + 1],
                           C_frag[i * 4 + p][j * 4 + 2],
                           C_frag[i * 4 + p][j * 4 + 3],
                           C_sts_addr + p * 8 * sizeof(float4));
                }

                __syncthreads();

#pragma unroll
                for (int p = 0; p < 16; ++p) {
                    stg32(C_lds_ptr[p * 32],
                          C_stg_ptr + (i * 16 + p) * n + j * 32,
                          j * 32 < n_guard);
                }
            }
        }
    } else {
#pragma unroll
        for (int i = 0; i < 2; ++i) {
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                StgFrag stg_frag(C_frag, j, i);

                C_tile_wb(stg_frag,
                          C_stg_ptr + i * 16 * n + j * 32,
                          C_lds_ptr,
                          C_sts_addr,
                          m,
                          n,
                          m_idx + i * 16,
                          n_idx + j * 32);
            }
        }
    }
}

static cudaError_t sgemm_128x128x8(int M, int N, int K, float const *A, float const *B, float *C) {
    dim3 grid((N + 127) / 128, (M + 127) / 128);
    sgemm_128x128x8_kernel<<<grid, 256>>>(A,B,C, M, N, K, K * sizeof(float), N * sizeof(float) * 8);
    return cudaSuccess;
}

/*
 * params: N, M, K, A, B, C
 */
DLL void MatmulOpt(int num_args, int *arg_types, void **args) {
    assert(num_args == 6);
    assert(arg_types[0] == INT32);
    int M = *static_cast<int *>(args[0]);
    assert(arg_types[1] == INT32);
    int N = *static_cast<int *>(args[1]);
    assert(arg_types[2] == INT32);
    int K = *static_cast<int *>(args[2]);
    assert(arg_types[3] == POINTER);
    auto *A = static_cast<float *>(args[3]);
    assert(arg_types[4] == POINTER);
    auto *B = static_cast<float *>(args[4]);
    assert(arg_types[5] == POINTER);
    auto *C = static_cast<float *>(args[5]);

    sgemm_128x128x8(M, N, K, A, B, C);
}

