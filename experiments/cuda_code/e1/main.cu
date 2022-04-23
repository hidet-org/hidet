#include <cuda.h>
#include <iostream>

inline __device__ unsigned get_lane_id() {
    unsigned ret;
    asm volatile("mov.u32 %0, %laneid;" : "=r"(ret));
    return ret;
}

inline __device__ unsigned get_warp_id() {
    unsigned ret;
    asm volatile("mov.u32 %0, %warpid;" : "=r"(ret));
    return ret;
}

__global__ void kernel() {
    const int actual_warpid = get_warp_id();
    const int actual_laneid = get_lane_id();
    printf("tid %i %i ctaid %i %i physical laneid %i  physical warpid %i\n", threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, actual_laneid, actual_warpid);
}

int main(int argc, char const *argv[]) {
    dim3 grid(2, 2);
    dim3 block(2, 2);

    kernel<<<grid, block>>>();
    cudaDeviceSynchronize();
    return 0;
}

/*
 * output:
 *   tid 0 0 ctaid 1 1 physical laneid 0  physical warpid 0
 *   tid 1 0 ctaid 1 1 physical laneid 1  physical warpid 0
 *   tid 0 1 ctaid 1 1 physical laneid 2  physical warpid 0
 *   tid 1 1 ctaid 1 1 physical laneid 3  physical warpid 0
 *   tid 0 0 ctaid 0 1 physical laneid 0  physical warpid 0
 *   tid 1 0 ctaid 0 1 physical laneid 1  physical warpid 0
 *   tid 0 1 ctaid 0 1 physical laneid 2  physical warpid 0
 *   tid 1 1 ctaid 0 1 physical laneid 3  physical warpid 0
 *   tid 0 0 ctaid 0 0 physical laneid 0  physical warpid 0
 *   tid 1 0 ctaid 0 0 physical laneid 1  physical warpid 0
 *   tid 0 1 ctaid 0 0 physical laneid 2  physical warpid 0
 *   tid 1 1 ctaid 0 0 physical laneid 3  physical warpid 0
 *   tid 0 0 ctaid 1 0 physical laneid 0  physical warpid 0
 *   tid 1 0 ctaid 1 0 physical laneid 1  physical warpid 0
 *   tid 0 1 ctaid 1 0 physical laneid 2  physical warpid 0
 *   tid 1 1 ctaid 1 0 physical laneid 3  physical warpid 0
 *
 *  indicates:
 *  the threadIdx in the same block is mapped to physical warp compactly.
 *  If we have blockDim.x % 32 != 0 and blockDim.y != 1, we should not use threadIdx.x & 32 to get lane id.
 */