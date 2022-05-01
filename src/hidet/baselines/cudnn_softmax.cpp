#include <map>
#include "cudnn_common.h"
#include <hidet/packedfunc.h>
#include <hidet/cuda_utils.h>

struct SoftmaxWorkload {
    int n;
    int c;
    int h;
    int w;
    SoftmaxWorkload(int n, int c, int h, int w):n(n), c(c), h(h), w(w){}
    bool operator<(const SoftmaxWorkload &rhs) const {
        const SoftmaxWorkload& lhs = *this;
        if(lhs.n != rhs.n)
            return lhs.n < rhs.n;
        if(lhs.c != rhs.c)
            return lhs.c < rhs.c;
        if(lhs.h != rhs.h)
            return lhs.h < rhs.h;
        if(lhs.w != rhs.w)
            return lhs.w < rhs.w;
        return false;
    }
};

struct CudnnSoftmaxContext {
    cudnnTensorDescriptor_t x_desc = nullptr;
    cudnnTensorDescriptor_t y_desc = nullptr;

    CudnnSoftmaxContext() {}

    explicit CudnnSoftmaxContext(const SoftmaxWorkload &workload) {
        CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
        CUDNN_CALL(cudnnSetTensor4dDescriptor(x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, workload.n, workload.c, workload.h, workload.w));
        y_desc = x_desc;
    }
};

void cudnn_softmax(const SoftmaxWorkload &workload, float* x, float* y) {
    static std::map<SoftmaxWorkload, CudnnSoftmaxContext> workload2ctx;
    auto iter = workload2ctx.find(workload);
    if(iter == workload2ctx.end()) {
        workload2ctx[workload] = CudnnSoftmaxContext(workload);
        iter = workload2ctx.find(workload);
    }
    const auto & ctx = iter->second;
    float alpha = 1.0;
    float beta = 0.0;
    CUDNN_CALL(cudnnSoftmaxForward(CudnnContext::global()->handle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL, &alpha, ctx.x_desc, x, &beta, ctx.y_desc, y));
}

// args: m, n, x, y
DLL void SoftmaxCudnn(int num_args, const int *arg_types, void **args) {
    assert(num_args == 6);
    for(int i = 0; i < 4; i++) {
        assert(arg_types[i] == INT32);
    }
    for(int i = 4; i < 4 + 2; i++) {
        assert(arg_types[i] == POINTER);
    }

    SoftmaxWorkload workload(INT_ARG(args[0]), INT_ARG(args[1]), INT_ARG(args[2]), INT_ARG(args[3]));
    auto *x = (float*)args[4];
    auto *y = (float*)args[5];
    cudnn_softmax(workload, x, y);
}
