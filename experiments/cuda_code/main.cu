#include <cassert>
#include <cuda_runtime.h>
#include <cstdio>

__global__ void kernel() {
    if (threadIdx.x / 2 == 0) {
        __syncthreads();
//        if(threadIdx.x % 2 == 0) {
            printf("A");
//        }
    } else {
//        if(threadIdx.x % 2 == 0) {
            printf("B");
//        }
        __syncthreads();
    }
}

int main() {
    kernel<<<1, 2>>>();
    cudaDeviceSynchronize();
}

