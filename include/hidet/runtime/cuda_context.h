#pragma once
#include <hidet/runtime/common.h>
#include <hidet/runtime/callbacks.h>
#include <hidet/runtime/context.h>
// #include <cuda_runtime.h>

struct CudaContext: BaseContext {
    /* The cuda stream the kernels will be launched on. */
    void* stream = nullptr;

    /**
     * Get the instance of cuda context.
     */
    static CudaContext* global();
};

/**
 * Set the cuda stream of cuda context.
 */
DLL void set_cuda_stream(void* stream);

/**
 * Get the cuda stream of cuda context.
 */
DLL void* get_cuda_stream();

/**
 * Request a workspace.
 */
DLL void* request_cuda_workspace(size_t nbytes, bool require_clean);

