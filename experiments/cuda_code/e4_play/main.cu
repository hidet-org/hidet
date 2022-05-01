#include <cstdio>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

__device__ void test() {
    printf("Hello World\n");
}

__global__ void kernel() {
    printf("Hello World\n");
    test();
}

int main() {
    kernel<<<1, 1>>>();
}