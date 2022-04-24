#include <cuda.h>

__global__ void kernel(int n, float *a, float *b) {
    auto id = threadIdx.x + blockIdx.x * blockDim.x;
    auto stride = blockDim.x * gridDim.x;
//    b[0] = a[0] / 0.55f;
    for(auto i = id; i < n; i += stride) {
//        b[i] = a[i] * 3;
//        b[i] = a[i] * 0.25f;
//        b[i] = a[i] * 0.55f;
//        b[i] = a[i] / 2;
//        b[i] = a[i] / 4;
//        b[i] = a[i] / 0.25f;
//        b[i] = a[i] / 0.55f;
//        b[i] = __fdividef(a[i],  0.55f);
        b[i] = a[i] * 0.0f;
    }
}

int main(int argc, char const *argv[]) {
}

/*
 *
        b[i] = a[i] * 3;
         FMUL R7, R2, 3 ;
        b[i] = a[i] * 0.25;
         FMUL R11, R2, 0.25 ;
        b[i] = a[i] * 0.55;
         F2F.F64.F32 R2, R6 ;
         DMUL R2, R2, c[0x2][0x0] ;
         F2F.F32.F64 R3, R2 ;
        b[i] = a[i] / 2;
         FMUL R13, R10, 0.5 ;
        b[i] = a[i] / 4;
         FMUL R15, R7, 0.25 ;
        b[i] = a[i] / 0.25;
         FMUL R17, R7, 4 ;
        b[i] = a[i] / 0.55;
    a float multiply a 0.0f will not be optimized into zero, might because there is special case such as NaN.
 */



