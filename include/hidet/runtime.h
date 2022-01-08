#pragma
#include <cuda_runtime.h>
#include <iostream>

#define CUDA_CALL(func) {                                       \
    cudaError_t e = (func);                                     \
    if(e != cudaSuccess) {                                      \
        std::cerr << __FILE__ << ": " << __LINE__ << ":"         \
        << "CUDA: " << cudaGetErrorString(e) << std::endl;      \
    }}                                                          \

