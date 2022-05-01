#pragma once

#include <iostream>
#include <map>
#include <cstring>
#include <cudnn.h>
#include "hidet/cuda_utils.h"


struct CudnnContext {
    cudnnHandle_t handle = nullptr;
    size_t workspace_bytes = 0;
    CudnnContext() {
        CUDNN_CALL(cudnnCreate(&handle));
    }
    void* get_workspace(size_t required_bytes) {
        if(required_bytes > workspace_bytes) {
            if(workspace != nullptr) {
                CUDA_CALL(cudaFree(workspace));
            }
            CUDA_CALL(cudaMalloc(&workspace, required_bytes));
            workspace_bytes = required_bytes;
        }
        return workspace;
    }
    static CudnnContext* global() {
        static CudnnContext ctx;
        return &ctx;
    }
private:
    void* workspace = nullptr;
};

