#include <hidet/packedfunc.h>
#include <cutlass/gemm/device/gemm.h>


cudaError_t gemm(int M, int N, int K, float alpha, float const *A, int lda, float const *B, int ldb, float beta, float *C, int ldc) {
    using ColumnMajor = cutlass::layout::ColumnMajor;
    using CutlassGemm = cutlass::gemm::device::Gemm<float, ColumnMajor, float, ColumnMajor, float, ColumnMajor>;
    CutlassGemm gemm_operator;

    CutlassGemm::Arguments args({M, N, K}, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta});

    cutlass::Status status = gemm_operator(args);

    if (status != cutlass::Status::kSuccess) {
        return cudaErrorUnknown;
    }

    return cudaSuccess;
}

/*
 * params: N, M, K, A, B, C
 */
DLL void MatmulCutlass(int num_args, int *arg_types, void **args) {
    assert(num_args == 6);
    assert(arg_types[0] == INT32);
    int M = *static_cast<int *>(args[0]);
    assert(arg_types[1] == INT32);
    int N = *static_cast<int *>(args[1]);
    assert(arg_types[2] == INT32);
    int K = *static_cast<int *>(args[2]);
    assert(arg_types[3] == FLOAT32);
    auto *A = static_cast<float *>(args[3]);
    assert(arg_types[4] == FLOAT32);
    auto *B = static_cast<float *>(args[4]);
    assert(arg_types[5] == FLOAT32);
    auto *C = static_cast<float *>(args[5]);

    gemm(M, N, K, 1.0f, A, M, B, K, 0.0f, C, M);
}


