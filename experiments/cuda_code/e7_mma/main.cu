#include <cstdio>
#include <cuda.h>
#include <functional>
#include <cuda_fp16.h>

#define CUDA_CALL(func) {                                                \
    cudaError_t e = (func);                                              \
    if(e != cudaSuccess) {                                               \
        cudaGetLastError();                                              \
        fprintf(stderr, "%s:%d: CUDA Error: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(1);                                                         \
    }                                                                    \
}

struct Tensor {
    int m;
    int n;
    int size;
    int nbytes;
    float *host_addr = nullptr;
    float *addr = nullptr;
    char layout = 'R';  // 'R' => row major layout; 'C' => column major layout

    Tensor(int m, int n, char layout = 'R'):m(m), n(n), layout(layout) {
        size = m * n;
        nbytes = m * n * sizeof(float);
        host_addr = new float[size];
        CUDA_CALL(cudaMalloc(&addr, nbytes))
    }

    Tensor(const Tensor&) = delete;

    Tensor& sync_host() {
        CUDA_CALL(cudaMemcpy(addr, host_addr, nbytes, cudaMemcpyKind::cudaMemcpyHostToDevice));
        return *this;
    }

    Tensor& sync_device() {
        CUDA_CALL(cudaMemcpy(host_addr, addr, nbytes, cudaMemcpyKind::cudaMemcpyDeviceToHost));
        return *this;
    }

    Tensor& init_with(const std::function<float(int,int)>& func) {
        for(int i = 0; i < m; i++) {
            for(int j = 0; j < n; j++) {
                at(i, j) = func(i, j);
            }
        }
        sync_host();
        return *this;
    }

    Tensor& rand() {
        return init_with([](int i, int j) {
            return float(std::rand()) / float(RAND_MAX);
        });
    }

    Tensor& range(int start=0) {
        return init_with([&](int i, int j) {
            return float(start + i * n + j);
        });
    }

    Tensor& zeros() {
        return init_with([](int i, int j) {
            return 0.0;
        });
    }

    float& at(int i, int j) const {
        if(layout == 'R') {
            return host_addr[i * n + j];
        } else {
            return host_addr[i + j * m];
        }
    }

    void print_host(const char* head = "", int space=3, int precision=1) const {
        if(strlen(head) > 0) {
            printf("%s\n", head);
        }
        for(int i = 0; i < m; i++) {
            for(int j = 0; j < n; j++) {
                printf("%*.*f ", space, precision, at(i, j));
            }
            printf("\n");
        }
    }

    ~Tensor() {
        delete[] host_addr;
        CUDA_CALL(cudaFree(addr));
    }
};


__device__ __forceinline__ void hidet_mma_sync_aligned_row_row_m16n8k8_f32_f16_f16_f32(
        uint32_t * __restrict__ a,
        uint32_t * __restrict__ b,
        uint32_t * __restrict__ c) {
    asm ("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};"
    : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
    : "r"(a[0]), "r"(a[1]), "r"(b[0]));
}


__global__ void matmul_mma_m16n8k8(float* A, float *B, float *C) {
    const int kM = 16;
    const int kN = 8;
    const int kK = 8;
    half regs_a[4];
    half regs_b[2];
    float regs_c[4];

    int lane_id = (int)threadIdx.x % 32;

    if(lane_id == 0) {
        printf("A:\n");
        for(int i = 0; i < kM; i++) {
            for(int j = 0; j < kK; j++) {
                printf("%.1f ", A[i * kK + j]);
            }
            printf("\n");
        }

        printf("B:\n");
        for(int i = 0; i < kK; i++) {
            for(int j = 0; j < kN; j++) {
                printf("%.1f ", B[i * kN + j]);
            }
            printf("\n");
        }
    }

    // load a
    for(int p = 0; p < 4; p++) {
        int i = lane_id / 4 + (p / 2) * 8;
        int j = lane_id % 4 * 2 + p % 2;
        regs_a[p] = (half)A[i * kK + j];
        printf("tid %d regs_a[%d] = A[%d, %d] = %.1f\n", lane_id, p, i, j, A[i * kK + j]);
    }

    // load b
    for(int p = 0; p < 2; p++) {
        int i = lane_id % 4 * 2 + p;
        int j = lane_id / 4;
        regs_b[p] = (half)B[i * kN + j];
        printf("tid %d regs_b[%d] = %.1f\n", lane_id, p, B[i * kN + j]);
    }

    // init c
    for(int p = 0; p < 4; p++) {
        regs_c[p] = 0.0;
    }

    // mma
    hidet_mma_sync_aligned_row_row_m16n8k8_f32_f16_f16_f32(
            reinterpret_cast<uint32_t*>(regs_a),
            reinterpret_cast<uint32_t*>(regs_b),
            reinterpret_cast<uint32_t*>(regs_c)
    );

    // store c
    for(int p = 0; p < 4; p++) {
        int i = lane_id / 4 + (p / 2) * 8;
        int j = lane_id % 4 * 2 + p % 2;
        C[i * kN + j] = regs_c[p];
        printf("tid %d regs_c[%d] = %.1f\n", lane_id, p, regs_c[p]);
    }
}

void matmul_host(const Tensor&A, const Tensor& B, const Tensor &C) {
    for(int i = 0; i < C.m; i++) {
        for(int j = 0; j < C.n; j++) {
            C.at(i, j) = 0.0;
            for(int k = 0; k < A.n; k++) {
                C.at(i, j) += A.at(i, k) * B.at(k, j);
            }
        }
    }
}


int main() {
    int m = 16, n = 8, k = 8;
    Tensor A(m, k);
    Tensor B(k, n);
    Tensor C(m, n);
    Tensor C_host(m, n);
    A.init_with([](int i, int j) {return i;}).sync_host();
    B.init_with([](int i, int j) {return j;}).sync_host();
    C.zeros().sync_host();
    matmul_mma_m16n8k8<<<1, 32>>>(A.addr, B.addr, C.addr);
    CUDA_CALL(cudaDeviceSynchronize());
    matmul_host(A, B, C_host);
    A.sync_device().print_host("A");
    B.sync_device().print_host("B");
    C.sync_device().print_host("C", 5);
    C_host.print_host("C host", 5);
}
