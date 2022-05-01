#pragma once

#include <cstdlib>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <iostream>
#include <hidet/common.h>
#include <hidet/logging.h>


#define CUDA_CALL(func) {                                                \
    cudaError_t e = (func);                                              \
    if(e != cudaSuccess) {                                               \
        cudaGetLastError();                                              \
        throw HidetException(__FILE__, __LINE__, cudaGetErrorString(e)); \
    }}

#define CUBLAS_CALL(func) {                                              \
    cublasStatus_t e = (func);                                           \
    if(e != CUBLAS_STATUS_SUCCESS) {                                     \
        throw HidetException(__FILE__, __LINE__,                         \
           std::string("cutlass error with code") + std::to_string(e));  \
    }}

#define CUDNN_CALL(func) {                                          \
    cudnnStatus_t _status = (func);                                 \
    if(_status != CUDNN_STATUS_SUCCESS) {                           \
        throw HidetException(__FILE__, __LINE__,                    \
           cudnnGetErrorString(_status));                           \
    }}

#define CURAND_CALL(func) {                                              \
    curandStatus_t e = (func);                                           \
    if(e != CURAND_STATUS_SUCCESS) {                                     \
        throw HidetException(__FILE__, __LINE__,                         \
           std::string("curand error with code") + std::to_string(e));   \
    }}


