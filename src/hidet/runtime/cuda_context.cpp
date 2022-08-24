#include <hidet/runtime/cuda_context.h>

CudaContext *CudaContext::global() {
    static thread_local CudaContext instance;
    return &instance;
}

DLL void set_cuda_stream(cudaStream_t stream) {
    CudaContext::global()->stream = stream;
}

DLL cudaStream_t get_cuda_stream() {
    return CudaContext::global()->stream;
}
