#include <iostream>
#include <map>
#include <cstring>
#include <cudnn.h>

#define CUDA_CALL(func) {                                           \
    cudaError_t e = (func);                                         \
    if(e != cudaSuccess) {                                          \
        std::cerr << __FILE__ << ": " << __LINE__ << ":"            \
        << "CUDA: " << cudaGetErrorString(e) << std::endl;          \
    }}

#define CUDNN_CALL(func) {                                          \
    cudnnStatus_t _status = (func);                                 \
    if(_status != CUDNN_STATUS_SUCCESS) {                           \
        std::cerr << __FILE__ << ": " << __LINE__ << ":"            \
        << "CUDNN: " << cudnnGetErrorString(_status) << std::endl;  \
    }}


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

