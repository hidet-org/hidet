#include <map>
#include <hidet/cudnn_common.h>
#include <hidet/packedfunc.h>
#include <hidet/runtime.h>

struct SoftmaxWorkload {
    int m;
    int n;
    SoftmaxWorkload(int m, int n):m(m), n(n){}
    bool operator<(const SoftmaxWorkload &rhs) const {
        const SoftmaxWorkload& lhs = *this;
        return lhs.m < rhs.m || (lhs.m == rhs.m && lhs.n < rhs.n);
    }
};

struct CudnnSoftmaxContext {
    cudnnTensorDescriptor_t x_desc = nullptr;
    cudnnTensorDescriptor_t y_desc = nullptr;

    CudnnSoftmaxContext() {}

    explicit CudnnSoftmaxContext(const SoftmaxWorkload &workload) {
        CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
        CUDNN_CALL(cudnnSetTensor4dDescriptor(x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, workload.m, workload.n, 1, 1));
        y_desc = x_desc;
    }
};

void cudnn_softmax(const SoftmaxWorkload &workload, float* x, float* y) {
    static std::map<SoftmaxWorkload, CudnnSoftmaxContext> workload2ctx;
    auto iter = workload2ctx.find(workload);
    if(iter == workload2ctx.end()) {
        workload2ctx[workload] = CudnnSoftmaxContext(workload);
    }
    const auto & ctx = workload2ctx[workload];
    float alpha = 1.0;
    float beta = 0.0;
    CUDNN_CALL(cudnnSoftmaxForward(CudnnContext::global()->handle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL, &alpha, ctx.x_desc, x, &beta, ctx.y_desc, y));
}

// args: m, n, x, y
DLL void SoftmaxCudnn(int num_args, const int *arg_types, void **args) {
    assert(num_args == 4);
    for(int i = 0; i < 2; i++) {
        assert(arg_types[i] == INT32);
    }
    for(int i = 2; i < 2 + 2; i++) {
        assert(arg_types[i] == POINTER);
    }

    SoftmaxWorkload workload(INT_ARG(args[0]), INT_ARG(args[1]));
    auto *x = (float*)args[2];
    auto *y = (float*)args[3];
    cudnn_softmax(workload, x, y);
}

