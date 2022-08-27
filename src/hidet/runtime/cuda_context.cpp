#include <hidet/runtime/cuda_context.h>
#include "hidet/cuda_utils.h"

void Workspace::reserve(size_t nbytes) {
    if(nbytes > this->allocated_nbytes) {
        if(base) {
            free_cuda_storage(reinterpret_cast<uint64_t>(this->base));
        }
        this->base = reinterpret_cast<void*>(allocate_cuda_storage(nbytes));
        CUDA_CALL(cudaMemsetAsync(this->base, 0, nbytes, CudaContext::global()->stream));
    }
}
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

DLL void* request_workspace(size_t nbytes, bool require_clean) {
    auto ctx = CudaContext::global();
    if(require_clean) {
        ctx->clean_workspace.reserve(nbytes);
        return ctx->clean_workspace.base;
    } else {
        ctx->dirty_workspace.reserve(nbytes);
        return ctx->dirty_workspace.base;
    }
}
