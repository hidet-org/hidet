#include <cstdio>
#include <exception>
#include <cuda_runtime.h>
#include "gemm.h"

#define CUDA_CALL(func) {                                                \
    cudaError_t e = (func);                                              \
    if(e != cudaSuccess) {                                               \
        cudaGetLastError();                                              \
        fprintf(stderr, "%s:%d: CUDA Error: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(1);                                                         \
    }                                                                    \
}

template<
        int M=1,
        int N=1,
        int K=1
>
struct GemmShape {
    static const int kM = M;
    static const int kN = N;
    static const int kK = K;
};


struct TensorRef {
    int stride;
    float *base;

    __device__
    float &operator()(int i, int j) const {
        return *(base + i * stride + j);
    }
};


struct Tensor {
    int m;
    int n;
    float *base;

    Tensor(int m, int n):m(m), n(n) {
        int size = m * n;
        base = nullptr;
        CUDA_CALL(cudaMalloc(&base, size))
    }

    Tensor(const Tensor&) = delete;

    ~Tensor() {
        CUDA_CALL(cudaFree(base));
    }

    TensorRef ref() const {
        return TensorRef{n, base};
    }
};


__global__ void gemm_reference(int M, int N, int K, TensorRef A, TensorRef B, TensorRef C) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if(i < M && j < N) {
        C(i, j) = 0.0;
        for(int k = 0; k < K; k++) {
            C(i, j) += A(i, k) * B(k, j);
        }
    }
}


template<
        typename BlockShape,
        typename WarpShape
        >
__global__ void gemm_simt(int M, int N, int K, TensorRef A, TensorRef B, TensorRef C) {
    __shared__ float smem_a[BlockShape::kM][BlockShape::kK];
    __shared__ float smem_b[BlockShape::kK][BlockShape::kN];
    float regs_c[BlockShape::kM][BlockShape::kN];

    for(int i = 0; i < BlockShape::kM; i++) {
        for(int j = 0; j < BlockShape::kN; j++) {
            regs_c[i][j] = 0.0;
        }
    }

    int block_x = blockIdx.x * BlockShape::kM;
    int block_y = blockIdx.y * BlockShape::kN;
    int warp_id = int(threadIdx.x) / 32;
    int lane_id = int(threadIdx.x) % 32;

    static_assert(BlockShape::kM % WarpShape::kM == 0, "");
    static_assert(BlockShape::kN % WarpShape::kN == 0, "");
    static_assert(BlockShape::kN % 32 == 0, "invariant violated.");

    constexpr int warps = (BlockShape::kM / WarpShape::kM) * (BlockShape::kN /  WarpShape::kN);
    constexpr int threads = warps * 32;


    int k_tiles = (K + BlockShape::kK - 1) / BlockShape::kK;
    for(int k_tile = 0; k_tile < k_tiles; k_tile++) {
        // load A and B from global memory to shared memory

        // thread block scoped matrix multiply accumulate

    }

}

void launch_reference(const Tensor &A, const Tensor &B, const Tensor &C) {
    dim3 block(256, 256);
    dim3 grid((C.m + block.x - 1) / block.x, (C.n + block.y - 1) / block.y);
    gemm_reference<<<grid, block>>>(C.m, C.n, A.n, A.ref(), B.ref(), C.ref());
}


void bench_ours(int m, int n, int k) {
    Tensor A(m, k), B(k, n), C(m, n);
    launch_reference(A, B, C);
    printf("%d %d %d\n", m, n, k);
}
