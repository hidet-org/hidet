#pragma once

#include <cstdlib>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <iostream>

#define CUDA_CALL(func) {                                           \
    cudaError_t e = (func);                                         \
    if(e != cudaSuccess) {                                          \
        std::cerr << __FILE__ << ":" << __LINE__ << " : "           \
        << "CUDA: " << cudaGetErrorString(e) << std::endl;          \
        exit(-1);                                                   \
    }}

#define CUBLAS_CALL(func) {                                     \
    cublasStatus_t e = (func);                                  \
    if(e != CUBLAS_STATUS_SUCCESS) {                            \
        std::cerr << __FILE__ << ":" << __LINE__ << " : "       \
        << "CUBLAS: error code " << e << std::endl;             \
        exit(-1);                                               \
    }}

#define CUDNN_CALL(func) {                                          \
    cudnnStatus_t _status = (func);                                 \
    if(_status != CUDNN_STATUS_SUCCESS) {                           \
        std::cerr << __FILE__ << ": " << __LINE__ << " : "          \
        << "CUDNN: " << cudnnGetErrorString(_status) << std::endl;  \
        exit(-1);                                                   \
    }}

#define CURAND_CALL(func) {                                     \
    curandStatus_t e = (func);                                  \
    if(e != CURAND_STATUS_SUCCESS) {                            \
        std::cerr << __FILE__ << ":" << __LINE__ << " : "       \
        << "CURAND: error code " << e << std::endl;             \
        exit(-1);                                               \
    }}


#ifdef assert
#undef assert
#endif
#define assert(x) if(!(x)){                                        \
        std::cerr << __FILE__ << ":" << __LINE__ << ": "           \
        << #x << "failed" << std::endl;                            \
        exit(-1);                                                  \
}

#ifndef DLL
#define DLL extern "C" __attribute__((visibility("default")))
#endif
