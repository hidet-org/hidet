#include <hidet/runtime.h>

static __global__ void fill_value_kernel_float(float* dst, uint64_t num_elements, float fill_value) {
    auto stride = gridDim.x * blockDim.x;
    for(auto idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num_elements; idx += stride) {
        dst[idx] = fill_value;
    }
}

DLL void hidet_cuda_fill_value(uint64_t addr, uint64_t num_elements, float fill_value) {
    dim3 block(512);
    dim3 grid((num_elements + block.x - 1) / block.x);
    fill_value_kernel_float<<<grid, block>>>(reinterpret_cast<float*>(addr), num_elements, fill_value);
}

