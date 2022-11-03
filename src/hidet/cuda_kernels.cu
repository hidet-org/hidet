#include "hidet/cuda_utils.h"

template<typename dtype>
static __global__ void fill_value_kernel(dtype* dst, uint64_t num_elements, dtype fill_value) {
    auto stride = gridDim.x * blockDim.x;
    for(auto idx = blockIdx.x * blockDim.x + threadIdx.x; idx < num_elements; idx += stride) {
        dst[idx] = fill_value;
    }
}

template<typename dtype>
static void fill_value_generic(uint64_t addr, uint64_t num_elements, dtype fill_value) {
    dim3 block(512);
    dim3 grid((num_elements + block.x - 1) / block.x);
    fill_value_kernel<<<grid, block>>>(reinterpret_cast<dtype*>(addr), num_elements, fill_value);
}

DLL void hidet_cuda_fill_value_float32(uint64_t addr, uint64_t num_elements, float fill_value) {
    fill_value_generic(addr, num_elements, fill_value);
}

DLL void hidet_cuda_fill_value_int32(uint64_t addr, uint64_t num_elements, int32_t fill_value) {
    fill_value_generic(addr, num_elements, fill_value);
}

DLL void hidet_cuda_fill_value_int64(uint64_t addr, uint64_t num_elements, int64_t fill_value) {
    fill_value_generic(addr, num_elements, fill_value);
}
