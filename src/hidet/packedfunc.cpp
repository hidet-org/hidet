#include <cassert>
#include <cstdio>
#include <cuda_runtime.h>

#include <hidet/packedfunc.h>
#include <hidet/runtime.h>

extern "C" {

DLL void CallPackedFunc(PackedFunc func, void** args) {
    auto f = PackedFunc_t(func.func_pointer);
    f(func.num_args, func.arg_types, args);
}

DLL void ProfilePackedFunc(PackedFunc func, void** args, int warmup, int number, int repeat, float* results) {
    cudaEvent_t start, end;
    CUDA_CALL(cudaEventCreate(&start));
    CUDA_CALL(cudaEventCreate(&end));

    for(int i = 0; i < warmup; i++) {
        CallPackedFunc(func, args);
    }

    for(int i = 0; i < repeat; i++) {
        CUDA_CALL(cudaDeviceSynchronize());
        CUDA_CALL(cudaEventRecord(start));
        for(int j = 0; j < number; j++) {
            CallPackedFunc(func, args);
        }
        CUDA_CALL(cudaEventRecord(end));
        CUDA_CALL(cudaDeviceSynchronize());
        CUDA_CALL(cudaEventElapsedTime(results + i, start, end));  // results[i] in milliseconds.
    }

    CUDA_CALL(cudaEventDestroy(start));
    CUDA_CALL(cudaEventDestroy(end));
}

DLL void packed_func_sample(int num_args, int *arg_types, void** args) {
    assert(num_args == 1);
    int type_code = arg_types[0];
    assert(type_code == INT32);
    int* arg = static_cast<int *>(args[0]);
    printf("hello, world!\n%d\n", *arg);
}

}

