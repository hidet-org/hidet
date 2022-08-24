#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <stdio.h>

__device__ int a;

__global__ void print_kernel() {
    if(threadIdx.x == 0) {
        printf("%d\n", a);
    }
}

int main() {
    print_kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
//    cudaDeviceSyncrhonize();
}
