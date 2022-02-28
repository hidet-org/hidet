#include <cassert>
#include <iostream>
#include <cublas_v2.h>
#include <hidet/packedfunc.h>


#define CUBLAS_CALL(func) {                                     \
    cublasStatus_t e = (func);                                  \
    if(e != CUBLAS_STATUS_SUCCESS) {                                      \
        std::cerr << __FILE__ << ": " << __LINE__ << ":"        \
        << "CUBLAS: error code " << e << std::endl;             \
    }}                                                          \


// live with the program, leaving the destroying to the driver
static cublasHandle_t cublas_handle = nullptr;


static cudaError_t cublas_sgemm_tc(int M, int N, int K, float const *A, float const *B, float *C) {
    if(cublas_handle == nullptr) {
        CUBLAS_CALL(cublasCreate(&cublas_handle));
        // Allow cublas to use tensor core when avaliable. This is the default behavior of cublas.
        // see 'https://docs.nvidia.com/cuda/cublas/index.html#tensorop-restrictions'
        CUBLAS_CALL(cublasSetMathMode(cublas_handle, CUBLAS_DEFAULT_MATH));
    }
    float alpha = 1.0;
    float beta = 0.0;
    CUBLAS_CALL(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, A,
                         M, B, K, &beta, C, M));

    return cudaSuccess;
}

/*
 * params: N, M, K, A, B, C
 */
DLL void MatmulCublasTc(int num_args, int *arg_types, void **args) {
    assert(num_args == 6);
    assert(arg_types[0] == INT32);
    int M = *static_cast<int *>(args[0]);
    assert(arg_types[1] == INT32);
    int N = *static_cast<int *>(args[1]);
    assert(arg_types[2] == INT32);
    int K = *static_cast<int *>(args[2]);
    assert(arg_types[3] == POINTER);
    auto *A = static_cast<float *>(args[3]);
    assert(arg_types[4] == POINTER);
    auto *B = static_cast<float *>(args[4]);
    assert(arg_types[5] == POINTER);
    auto *C = static_cast<float *>(args[5]);

    cublas_sgemm_tc(M, N, K, A, B, C);
}

