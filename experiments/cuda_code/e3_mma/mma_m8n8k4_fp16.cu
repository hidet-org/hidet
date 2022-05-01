#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include "e3.h"

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
unsigned &reg32(half *base, int idx) {
    return ((unsigned *)(base))[idx];
}

__device__ __forceinline__
void mma_m8n8k4_fp16(half a[4], half b[4], half c[8]) {
    asm volatile(
        "{"
        "mma.sync.aligned.m8n8k4.row.col.f16.f16.f16.f16 {%0, %1, %2, %3}, {%4, %5}, {%6, %7}, {%8, %9, %10, %11};"
        "}"
        : 
        "=r"(reg32(c, 0)), "=r"(reg32(c, 1)), "=r"(reg32(c, 2)), "=r"(reg32(c, 3))
        : 
        "r"(reg32(a, 0)), "r"(reg32(a, 1)), 
        "r"(reg32(b, 0)), "r"(reg32(b, 1)), 
        "r"(reg32(c, 0)), "r"(reg32(c, 1)), "r"(reg32(c, 2)), "r"(reg32(c, 3))
    );
}


__global__ void kernel(half ga[8][4], half gb[4][8], half gc[8][8]) {
    half a[4], b[4], c[8];

    int laneid = threadIdx.x % 32;
    if(threadIdx.x % 16 < 4) {
        for(int i = 0; i < 8; i++) {
            c[i] = 0.0;
        }
        for(int i = 0; i < 4; i++) {
            a[i] = ga[laneid % 4 + laneid / 16 * 4][i];
            b[i] = gb[i][laneid % 4 + laneid / 16 * 4];
        }
    }

    mma_m8n8k4_fp16(a, b, c);

    if(threadIdx.x % 16 < 4) {
        for(int i = 0; i < 8; i++) {
            gc[laneid % 4 + laneid / 16 * 4][i] = c[i];
        }
        // printf("%d: %f %f %f %f %f %f %f %f\n", threadIdx.x, __half2float(c[0]), __half2float(c[1]), __half2float(c[2]), __half2float(c[3]), __half2float(c[4]), __half2float(c[5]), __half2float(c[6]), __half2float(c[7]));
    }
}

void test_mma_m8n8k4_fp16() {
    half *aa, *bb, *cc;
    cudaMallocManaged(&aa, sizeof(half[8][4]));
    cudaMallocManaged(&bb, sizeof(half[4][8]));
    cudaMallocManaged(&cc, sizeof(half[8][8]));


    half (*a)[8][4] = reinterpret_cast<half(*)[8][4]>(aa);
    half (*b)[4][8] = reinterpret_cast<half(*)[4][8]>(bb);
    half (*c)[8][8] = reinterpret_cast<half(*)[8][8]>(cc);
    printf("a:\n");
    for(int i = 0; i < 8; i++) {
        for(int k = 0; k < 4; k++) {
            (*a)[i][k] = i;
            printf("%f ", float((*a)[i][k]));
        }
        printf("\n");
    }
    printf("b:\n");
    for(int k = 0; k < 4; k++) {
        for(int j = 0; j < 8; j++) {
            (*b)[k][j] = j;
            printf("%f ", float((*b)[k][j]));
        }
        printf("\n");
    }
    for(int i = 0; i < 8; i++) {
        for(int j = 0; j < 8; j++) {
            (*c)[i][j] = 0.0;
        }
    }
    kernel<<<1, 32>>>(*a, *b, *c);
    cudaDeviceSynchronize();
    printf("c:\n");
    for(int i = 0; i < 8; i++) {
        for(int j = 0; j < 8; j++) {
            printf("%f ", float((*c)[i][j]));
        }
        printf("\n");
    }
}




