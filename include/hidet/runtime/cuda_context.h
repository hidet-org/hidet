#pragma once
#include <hidet/common.h>
#include <hidet/runtime/callbacks.h>
#include <cuda_runtime.h>

struct Workspace {
    void* base;
    size_t allocated_nbytes;
    Workspace() {
        base = nullptr;
        allocated_nbytes = 0;
    }
    void reserve(size_t nbytes);
};

struct CudaContext {
    cudaStream_t stream;
    Workspace clean_workspace;
    Workspace dirty_workspace;
    size_t workspace_nbytes;
    static CudaContext* global();
};

DLL void set_cuda_stream(cudaStream_t stream);

DLL cudaStream_t get_cuda_stream();

DLL void* request_workspace(size_t nbytes, bool require_clean);

