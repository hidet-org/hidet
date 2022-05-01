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

int main() {

    float *A, *B, *C;


}
