#include <map>
#include <hidet/cudnn_common.h>
#include <hidet/packedfunc.h>
#include <hidet/runtime.h>

struct Pool2dWorkload {
    int n;
    int c;
    int h;
    int w;
    int kx;
    int ky;
    int px;
    int py;
    int sx;
    int sy;
    int mode;
//    CUDNN_POOLING_MAX                           = 0,
//    CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING = 1, /* count for average includes padded values */
//    CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING = 2, /* count for average does not include padded values */
//    CUDNN_POOLING_MAX_DETERMINISTIC             = 3
    bool operator<(const Pool2dWorkload &rhs) const {
        return memcmp(this, &rhs, sizeof(Pool2dWorkload)) < 0;
    }
};

struct CudnnSoftmaxContext {
    cudnnPoolingDescriptor_t pool_desc = nullptr;
    cudnnPoolingMode_t pool_mode = cudnnPoolingMode_t(0);
    cudnnTensorDescriptor_t x_desc = nullptr;
    cudnnTensorDescriptor_t y_desc = nullptr;

    CudnnSoftmaxContext() = default;

    explicit CudnnSoftmaxContext(const Pool2dWorkload &workload) {
        int out_n, out_c, out_h, out_w;
        pool_mode = cudnnPoolingMode_t(workload.mode);
        CUDNN_CALL(cudnnCreatePoolingDescriptor(&pool_desc));
        CUDNN_CALL(cudnnSetPooling2dDescriptor(pool_desc, pool_mode, CUDNN_NOT_PROPAGATE_NAN, workload.kx, workload.ky, workload.px, workload.py, workload.sx, workload.sy));
        CUDNN_CALL(cudnnCreateTensorDescriptor(&x_desc));
        CUDNN_CALL(cudnnSetTensor4dDescriptor(x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, workload.n, workload.c, workload.h, workload.w));
        CUDNN_CALL(cudnnCreateTensorDescriptor(&y_desc));
        CUDNN_CALL(cudnnGetPooling2dForwardOutputDim(pool_desc, x_desc, &out_n, &out_c, &out_h, &out_w));
        CUDNN_CALL(cudnnSetTensor4dDescriptor(y_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, out_n, out_c, out_h, out_w));
        assert(out_n == workload.n);
        assert(out_c == workload.c);
        assert(out_h == (workload.h - workload.kx + 2 * workload.px) / workload.sx + 1);
        assert(out_w == (workload.w - workload.ky + 2 * workload.py) / workload.sy + 1);
    }
};

void cudnn_pool2d(const Pool2dWorkload &workload, float* x, float* y) {
    static std::map<Pool2dWorkload, CudnnSoftmaxContext> workload2ctx;
    auto iter = workload2ctx.find(workload);
    if(iter == workload2ctx.end()) {
        workload2ctx[workload] = CudnnSoftmaxContext(workload);
        iter = workload2ctx.find(workload);
    }
    const auto & ctx = iter->second;
    const float alpha = 1.0;
    const float beta = 0.0;
    CUDNN_CALL(cudnnPoolingForward(CudnnContext::global()->handle, ctx.pool_desc, &alpha, ctx.x_desc, x, &beta, ctx.y_desc, y));
}

// args: n, c, h, w, kx, ky, px, py, sx, sy, mode, x, y
DLL void Pool2dCudnn(int num_args, const int *arg_types, void **args) {
    assert(num_args == 13);
    for(int i = 0; i < 11; i++) {
        assert(arg_types[i] == INT32);
    }
    for(int i = 11; i < 11 + 2; i++) {
        assert(arg_types[i] == POINTER);
    }

    Pool2dWorkload workload{INT_ARG(args[0]), INT_ARG(args[1]), INT_ARG(args[2]), INT_ARG(args[3]), INT_ARG(args[4]),
                            INT_ARG(args[5]), INT_ARG(args[6]), INT_ARG(args[7]), INT_ARG(args[8]), INT_ARG(args[9]), INT_ARG(args[10])};
    auto *x = (float*)args[11];
    auto *y = (float*)args[12];
    cudnn_pool2d(workload, x, y);
}
