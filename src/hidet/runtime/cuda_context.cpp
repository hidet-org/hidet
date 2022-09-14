#include <hidet/runtime/cuda_context.h>
#include "hidet/cuda_utils.h"


CudaContext *CudaContext::global() {
    static CudaContext instance;
    return &instance;
}

static void reserve_cuda_workspace(Workspace &workspace, size_t nbytes) {
    if(nbytes > workspace.allocated_nbytes) {
        if(workspace.base) {
            free_cuda_storage(reinterpret_cast<uint64_t>(workspace.base));
        }
        workspace.base = reinterpret_cast<void*>(allocate_cuda_storage(nbytes));
        if(workspace.base == nullptr) {
            throw HidetException(__FILE__, __LINE__, "allocate workspace failed.");
        }
        CUDA_CALL(cudaMemsetAsync(workspace.base, 0, nbytes, CudaContext::global()->stream));
    }
}

DLL void set_cuda_stream(cudaStream_t stream) {
    CudaContext::global()->stream = stream;
}

DLL cudaStream_t get_cuda_stream() {
    return CudaContext::global()->stream;
}

DLL void* request_cuda_workspace(size_t nbytes, bool require_clean) {
    API_BEGIN()
        auto ctx = CudaContext::global();
        if(require_clean) {
            reserve_cuda_workspace(ctx->clean_workspace, nbytes);
            return ctx->clean_workspace.base;
        } else {
            reserve_cuda_workspace(ctx->dirty_workspace, nbytes);
            return ctx->dirty_workspace.base;
        }
    API_END(nullptr)
}
