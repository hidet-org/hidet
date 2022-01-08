#include <cassert>
#include <hidet/runtime.h>
#include <hidet/packedfunc.h>

/// Reference GEMM computation.
static __global__ void gemm_kernel(int M, int N, int K, float alpha, float const *A, int lda, float const *B, int ldb, float beta, float *C, int ldc) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if (i < M && j < N) {
        float accumulator = 0;

        for (int k = 0; k < K; ++k) {
            accumulator += A[i + k * lda] * B[k + j * ldb];
        }

        C[i + j * ldc] = alpha * accumulator + beta * C[i + j * ldc];
    }
}


/// Reference GEMM computation.
static cudaError_t gemm(int M, int N, int K, float alpha, float const *A, int lda, float const *B, int ldb, float beta, float *C, int ldc) {
    dim3 block(16, 16);
    dim3 grid((M + block.x - 1) / block.x,
              (N + block.y - 1) / block.y);

    gemm_kernel<<<grid, block>>>(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);

    return cudaSuccess;
}

/*
 * params: N, M, K, A, B, C
 */
DLL void MatmulReference(int num_args, int *arg_types, void **args) {
    assert(num_args == 6);
    assert(arg_types[0] == INT32);
    int M = *static_cast<int *>(args[0]);
    assert(arg_types[1] == INT32);
    int N = *static_cast<int *>(args[1]);
    assert(arg_types[2] == INT32);
    int K = *static_cast<int *>(args[2]);
    assert(arg_types[3] == FLOAT32);
    auto *A = static_cast<float *>(args[3]);
    assert(arg_types[4] == FLOAT32);
    auto *B = static_cast<float *>(args[4]);
    assert(arg_types[5] == FLOAT32);
    auto *C = static_cast<float *>(args[5]);

    gemm(M, N, K, 1.0f, A, M, B, K, 0.0f, C, M);
}
