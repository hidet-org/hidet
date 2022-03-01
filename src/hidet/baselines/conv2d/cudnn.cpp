#include <map>
#include <cstring>
#include <cudnn.h>
#include <hidet/runtime.h>
#include <hidet/packedfunc.h>

struct CudnnContext {
    cudnnHandle_t handle = nullptr;
    size_t workspace_bytes = 0;
    void* workspace = nullptr;
    CudnnContext() {
        CUDNN_CALL(cudnnCreate(&handle));
    }
    void* get_workspace(size_t required_bytes) {
        if(required_bytes > workspace_bytes) {
            if(workspace != nullptr) {
                CUDA_CALL(cudaFree(workspace));
            }
            CUDA_CALL(cudaMalloc(&workspace, required_bytes));
            workspace_bytes = required_bytes;
        }
        return workspace;
    }
    static CudnnContext* global() {
        static CudnnContext ctx;
        return &ctx;
    }
};

struct Conv2dSetting {
    int batch_size;
    int in_channels;
    int height;
    int width;
    int out_channels;
    int kernel_h;
    int kernel_w;
    int padding_h;
    int padding_w;
    int stride_h;
    int stride_w;
    int math_mode;
    int algo;

    // math_mode:
    // 0 - default math
    // 1 - allow use tensor core
    // 2 - allow use tensor core, actively down conversion
    // 3 - only fma instruction allowed (do not allow tensor core)
    // see https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnMathType_t

    // algo:
    // CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM         = 0,
    // CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM = 1,
    // CUDNN_CONVOLUTION_FWD_ALGO_GEMM                  = 2,
    // CUDNN_CONVOLUTION_FWD_ALGO_DIRECT                = 3,
    // CUDNN_CONVOLUTION_FWD_ALGO_FFT                   = 4,
    // CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING            = 5,
    // CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD              = 6,
    // CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED     = 7,
    bool operator<(const Conv2dSetting &rhs) const {
        const Conv2dSetting &lhs = *this;
        // only used when this class has only integers
        return std::memcmp(&lhs, &rhs, sizeof(Conv2dSetting)) < 0;
    }
};


struct Conv2dContext {
    cudnnConvolutionDescriptor_t conv_desc;
    cudnnConvolutionFwdAlgo_t conv_algo;
    cudnnTensorDescriptor_t x_desc;
    cudnnFilterDescriptor_t w_desc;
    cudnnTensorDescriptor_t y_dest;
};

Conv2dContext create_conv2d_context(const Conv2dSetting &setting) {
    //  assume NCHW format, float32 data type for now
    cudnnTensorFormat_t data_layout = CUDNN_TENSOR_NCHW;
    cudnnDataType_t data_type = CUDNN_DATA_FLOAT;

    // handle
    Conv2dContext ctx{};

    // x desc
    CUDNN_CALL(cudnnCreateTensorDescriptor(&ctx.x_desc));
    CUDNN_CALL(cudnnSetTensor4dDescriptor(ctx.x_desc, data_layout, data_type, setting.batch_size, setting.in_channels, setting.height, setting.width));

    // w desc
    CUDNN_CALL(cudnnCreateFilterDescriptor(&ctx.w_desc));
    CUDNN_CALL(cudnnSetFilter4dDescriptor(ctx.w_desc, data_type, data_layout, setting.out_channels, setting.in_channels, setting.kernel_h, setting.kernel_w));

    // conv desc
    CUDNN_CALL(cudnnCreateConvolutionDescriptor(&ctx.conv_desc));
    CUDNN_CALL(cudnnSetConvolution2dDescriptor(ctx.conv_desc, setting.padding_h, setting.padding_w, setting.stride_h, setting.stride_w, 1, 1, CUDNN_CONVOLUTION, data_type));
    CUDNN_CALL(cudnnSetConvolutionMathType(ctx.conv_desc, cudnnMathType_t(setting.math_mode)));

    // y desc
    CUDNN_CALL(cudnnCreateTensorDescriptor(&ctx.y_dest));
    int y_n, y_c, y_h, y_w;
    CUDNN_CALL(cudnnGetConvolution2dForwardOutputDim(ctx.conv_desc, ctx.x_desc, ctx.w_desc, &y_n, &y_c, &y_h, &y_w));
    assert(y_n == setting.batch_size);
    assert(y_c == setting.out_channels);
    CUDNN_CALL(cudnnSetTensor4dDescriptor(ctx.y_dest, data_layout, data_type, setting.batch_size, setting.out_channels, y_h, y_w));

    // algo
    if(setting.algo == -1) {
        const int max_algo_num = 100;
        int ret_algo_num;
        cudnnConvolutionFwdAlgoPerf_t perf_results[max_algo_num];
        CUDNN_CALL(cudnnFindConvolutionForwardAlgorithm(CudnnContext::global()->handle, ctx.x_desc, ctx.w_desc, ctx.conv_desc, ctx.y_dest, max_algo_num, &ret_algo_num, perf_results));
        bool found = false;
        for(int i = 0; i < ret_algo_num; i++) {
            if(perf_results[i].status == CUDNN_STATUS_SUCCESS && perf_results[i].mathType == setting.math_mode) {
                found = true;
                ctx.conv_algo = perf_results[i].algo;
            }
        }
        if(!found) {
            std::cerr << "Have not found an algorithm for given math mode; use the default math mode";
            ctx.conv_algo = perf_results[0].algo;
        }
    } else {
        assert(0 <= setting.algo && setting.algo <= CUDNN_CONVOLUTION_FWD_ALGO_COUNT);
        ctx.conv_algo = cudnnConvolutionFwdAlgo_t(setting.algo);
    }

    return ctx;
}



static void cudnn_conv2d(const Conv2dSetting &setting, float *x, float *w, float *y) {
    static std::map<Conv2dSetting, Conv2dContext> setting2context;
    auto iter = setting2context.find(setting);
    if(iter == setting2context.end()) {
        setting2context[setting] = create_conv2d_context(setting);
        iter = setting2context.find(setting);
    }
    const Conv2dContext &ctx = iter->second;
    const float alpha = 1.0;
    const float beta = 0.0;
    auto cudnn_ctx = CudnnContext::global();
    CUDNN_CALL(cudnnConvolutionForward(cudnn_ctx->handle, &alpha, ctx.x_desc, x, ctx.w_desc, w, ctx.conv_desc, ctx.conv_algo,
                                       cudnn_ctx->workspace, cudnn_ctx->workspace_bytes, &beta, ctx.y_dest, y));
}

// params:
//  int batch_size
//  int in_channels
//  int height
//  int width
//  int out_channels
//  int kernel_h
//  int kernel_w
//  int padding_h
//  int padding_w
//  int stride_h
//  int stride_w
//  int algo
//  int math_mode
//  float* x
//  float* w
//  float* y
DLL void Conv2dCudnn(int num_args, const int *arg_types, void **args) {
    assert(num_args == 13 + 3);
    for(int i = 0; i < 13; i++) {
        assert(arg_types[i] == INT32);
    }
    for(int i = 13; i < 13 + 3; i++) {
        assert(arg_types[i] == POINTER);
    }

    Conv2dSetting setting{INT_ARG(args[0]), INT_ARG(args[1]), INT_ARG(args[2]), INT_ARG(args[3]),
                           INT_ARG(args[4]), INT_ARG(args[5]), INT_ARG(args[6]), INT_ARG(args[7]),
                           INT_ARG(args[8]), INT_ARG(args[9]), INT_ARG(args[10]), INT_ARG(args[11]), INT_ARG(args[12])};
    auto *x = (float*)args[13];
    auto *w = (float*)args[14];
    auto *y = (float*)args[15];
    cudnn_conv2d(setting, x, w, y);
}
