#pragma once
#include <hidet/common.h>
#include <cuda_runtime.h>


struct CudaContext {
    cudaStream_t stream;
    static CudaContext* global();
};

DLL void set_cuda_stream(cudaStream_t stream);

DLL cudaStream_t get_cuda_stream();
