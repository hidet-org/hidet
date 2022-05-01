#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>


__device__ __forceinline__
void wmma_load_a(half *p, half regs[16], int stride) {
    auto *r = reinterpret_cast<unsigned*>(regs);
    asm volatile(
    "{"
    "   wmma.load.a.sync.aligned.m16n16k16.row.f16 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8], %9;"
    "}"
    :
    "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]),
    "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7])
    :
    "l"(p), "r"(stride)
    );
}

__device__ __forceinline__
void wmma_load_b(half *p, half regs[16], int stride) {
    auto *r = reinterpret_cast<unsigned*>(regs);
    asm volatile(
    "{"
    "   wmma.load.b.sync.aligned.m16n16k16.row.f16 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8], %9;"
    "}"
    :
    "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]),
    "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7])
    :
    "l"(p), "r"(stride)
    );
}

__device__ __forceinline__
void swap(half &a, half &b) {
    half c;
    c = a;
    a = b;
    b = c;
}

__device__ __forceinline__
void wmma_mma(half a_regs[16], half b_regs[16], half c_regs[8]) {
    auto *a = reinterpret_cast<unsigned*>(a_regs);
    auto *b = reinterpret_cast<unsigned*>(b_regs);
    auto *c = reinterpret_cast<unsigned*>(c_regs);
    asm volatile(
    "{"
    "   wmma.mma.sync.aligned.row.row.m16n16k16.f16.f16 "
    "       {%0, %1, %2, %3}, "                     // d
    "       {%4, %5, %6, %7, %8, %9, %10, %11}, "   // a
    "       {%12, %13, %14, %15, %16, %17, %18, %19}, "   // b
    "       {%0, %1, %2, %3};"                      // c
    "}"
    :
    "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
    :
    "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(a[4]), "r"(a[5]), "r"(a[6]), "r"(a[7]),
    "r"(b[0]), "r"(b[1]), "r"(b[2]), "r"(b[3]), "r"(b[4]), "r"(b[5]), "r"(b[6]), "r"(b[7])
    );
}

__device__ __forceinline__
void wmma_store(half c_regs[8], half *p, int stride) {
    auto *r = reinterpret_cast<unsigned*>(c_regs);
    asm volatile(
    "{"
    "   wmma.store.d.sync.aligned.row.m16n16k16.f16 [%4], {%0, %1, %2, %3}, %5;"
    "}"
    ::
    "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]),
    "l"(p),
    "r"(stride)
    );
}

__global__ void  matmul_m16n16k16_fp16_kernel(half* A, half *B, half *C) {
    half regs_a[16];
    half regs_b[16];
    half regs_c[8];
    for(auto & c : regs_c) {
        c = 0.0;
    }
    wmma_load_a(A, regs_a, 16);
    wmma_load_b(B, regs_b, 16);
    printf("thread %d regs_a %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d\n",
           threadIdx.x,
           (int)regs_a[0],
           (int)regs_a[1],
           (int)regs_a[2],
           (int)regs_a[3],
           (int)regs_a[4],
           (int)regs_a[5],
           (int)regs_a[6],
           (int)regs_a[7],
           (int)regs_a[8],
           (int)regs_a[9],
           (int)regs_a[10],
           (int)regs_a[11],
           (int)regs_a[12],
           (int)regs_a[13],
           (int)regs_a[14],
           (int)regs_a[15]
    );
    printf("thread %d regs_b %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d\n",
           threadIdx.x,
           (int)regs_b[0],
           (int)regs_b[1],
           (int)regs_b[2],
           (int)regs_b[3],
           (int)regs_b[4],
           (int)regs_b[5],
           (int)regs_b[6],
           (int)regs_b[7],
           (int)regs_b[8],
            (int)regs_b[9],
            (int)regs_b[10],
            (int)regs_b[11],
            (int)regs_b[12],
            (int)regs_b[13],
            (int)regs_b[14],
            (int)regs_b[15]
    );
    printf("thread %d regs_c %d %d %d %d %d %d %d %d\n",
           threadIdx.x,
           (int)regs_c[0],
           (int)regs_c[1],
           (int)regs_c[2],
           (int)regs_c[3],
           (int)regs_c[4],
           (int)regs_c[5],
           (int)regs_c[6],
           (int)regs_c[7]
    );
    wmma_mma(regs_a, regs_b, regs_c);
    printf("thread %d regs_c %d %d %d %d %d %d %d %d\n",
           threadIdx.x,
           (int)regs_c[0],
           (int)regs_c[1],
           (int)regs_c[2],
           (int)regs_c[3],
           (int)regs_c[4],
           (int)regs_c[5],
           (int)regs_c[6],
           (int)regs_c[7]
    );
    wmma_store(regs_c, C, 16);
}

void matmul_m16n16k16_fp16(half* A, half* B, half* C) {
    matmul_m16n16k16_fp16_kernel<<<1, 32>>>(A, B, C);
}

void print(const char *name, half A[16][16]) {
    printf("%s\n", name);
    for(int i = 0; i < 16; i++) {
        for(int j = 0; j < 16; j++) {
            printf("%.1f ", (float)A[i][j]);
        }
        printf("\n");
    }
}

void test_m16n16k16_fp16() {
    half (*A)[16];
    half (*B)[16];
    half (*C)[16];
    cudaMallocManaged(&A, sizeof(half) * 16 * 16);
    cudaMallocManaged(&B, sizeof(half) * 16 * 16);
    cudaMallocManaged(&C, sizeof(half) * 16 * 16);

    for(int i = 0; i < 16; i++) {
        for(int j = 0; j < 16; j++) {
            A[i][j] = i;
        }
    }
    for(int i = 0; i < 16; i++) {
        for(int j = 0; j < 16; j++) {
            B[i][j] = j;
        }
    }
    for(int i = 0; i < 16; i++) {
        for(int j = 0; j < 16; j++) {
            C[i][j] = 0;
        }
    }
    print("A", A);
    print("B", B);
    matmul_m16n16k16_fp16((half*)A, (half*)B, (half*)C);
    cudaDeviceSynchronize();
    print("C", C);

}
