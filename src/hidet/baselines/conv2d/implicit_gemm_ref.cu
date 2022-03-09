#include <hidet/runtime.h>
#include <hidet/packedfunc.h>

// x[n, rc, h, w], w[c, rc, rx, ry], y[n, c, p, q]
// h = p * stride_h - padding_h + rx
// i -> n, p, q
// j -> c
// k -> rc, rx, ry
// i, k -> n, rc, h, w
// k, j -> c, rc, rx, ry
// i, j -> n, c, p, q
__global__ void conv2d_implicit_gemm_ref_kernel(
        int batch_size, int in_channels, int height, int width, int out_channels,
        int kernel_h, int kernel_w, int padding_h, int padding_w, int stride_h, int stride_w,
        const float* input, const float* weight, float *output) {
    int out_width = (width - kernel_h + 2 * padding_h) / stride_h + 1;
    int out_height = (height - kernel_w + 2 * padding_w) / stride_w + 1;
    int gemm_m_max = batch_size * out_height * out_width;
    int gemm_k_max = in_channels * kernel_h * kernel_w;
    int gemm_n_max = out_channels;

    int i = (int)(threadIdx.x + blockIdx.x * blockDim.x);
    int j = (int)(threadIdx.y + blockIdx.y * blockDim.y);
//    printf("i %d j %d m_max %d n_max %d\n", i, j, gemm_m_max, gemm_n_max);
    if(i >= gemm_m_max || j >= gemm_n_max) {
        return;
    }
    int n = i / (out_height * out_width);
    int c = j;
    int p = (i / out_width) % out_height;
    int q = i % out_height;
    float acc = 0.0f;
    for(int k = 0; k < gemm_k_max; k++) {
        int rc = k / (kernel_h * kernel_w);
        int rx = (k / kernel_w) % kernel_h;
        int ry = k % kernel_w;
        int h = p * stride_h - padding_h + rx;
        int w = q * stride_w - padding_w + ry;
//        printf("k h w %d %d %d", k, h, w);
        if(h < 0 || h >= height || w < 0 || w >= width) {
            continue;
        }
        int x_offset = n * in_channels * height * width + rc * height * width + h * width + w;
        int w_offset = c * in_channels * kernel_h * kernel_w + rc * kernel_h * kernel_w + rx * kernel_w + ry;
//        printf("%d %d %.1f %.1f\n", x_offset, w_offset, input[x_offset], weight[w_offset]);
        acc += input[x_offset] * weight[w_offset];
    }
    int y_offset = n * out_channels * out_width * out_height + c * out_width * out_height + p * out_width + q;
//    printf("out %d %.1f\n", y_offset, output[y_offset]);
    output[y_offset] = acc;
}


DLL void Conv2dImplicitGemmReference(int num_args, const int *arg_types, void **args) {
    assert(num_args == 11 + 3);
    for(int i = 0; i < 11; i++) {
        assert(arg_types[i] == INT32);
    }
    for(int i = 13; i < 11 + 3; i++) {
        assert(arg_types[i] == POINTER);
    }
    int batch_size = INT_ARG(args[0]);
    int in_channels = INT_ARG(args[1]);
    int height = INT_ARG(args[2]);
    int width = INT_ARG(args[3]);
    int out_channels = INT_ARG(args[4]);
    int kernel_h = INT_ARG(args[5]);
    int kernel_w = INT_ARG(args[6]);
    int padding_h = INT_ARG(args[7]);
    int padding_w = INT_ARG(args[8]);
    int stride_h = INT_ARG(args[9]);
    int stride_w = INT_ARG(args[10]);
    auto *in = (float*)args[11];
    auto *weight = (float*)args[12];
    auto *out = (float*)args[13];

    int out_height = (height + 2 * padding_h - kernel_h) / stride_h + 1;
    int out_width = (width + 2 * padding_w - kernel_w) / stride_w + 1;
    int gemm_m_max = batch_size * out_height * out_width;
    int gemm_n_max = out_channels;
    dim3 block(16, 16);
    dim3 grid((gemm_m_max + block.x - 1) / block.x, (gemm_n_max + block.y - 1) / block.y);
    conv2d_implicit_gemm_ref_kernel<<<grid, block>>>(batch_size, in_channels, height, width, out_channels, kernel_h,
                                                     kernel_w, padding_h, padding_w, stride_h, stride_w, in, weight, out);
}
