#pragma once
#include <hidet/common.h>
#include <cuda_runtime.h>

struct Workspace {
    void* base;
    size_t nbytes;
    void reserve(size_t nbytes) {

    }
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

