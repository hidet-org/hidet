#include <hidet/runtime.h>
#include <hidet/packedfunc.h>


static __global__ void conv2d_ref_kernel(
        int batch_size, int in_channels, int height, int width, int out_channels,
        int kernel_h, int kernel_w, int padding_h, int padding_w, int stride_h, int stride_w,
        const float* in, const float* weight, float *out
) {
    int out_height = (height + 2 * padding_h - kernel_h) / stride_h + 1;
    int out_width = (width + 2 * padding_w - kernel_w) / stride_w + 1;
    int in_offset[3] = {in_channels * height * width, height * width, width};
    int out_offset[3] = {out_channels * out_height * out_width, out_height * out_width, out_width};
    int weight_offset[3] = {in_channels * kernel_h * kernel_w, kernel_h * kernel_w, kernel_w};
#define ACCESS(arr, a, b, c, d, offset) (*((arr) + (a) * (offset)[0] + (b) * (offset)[1] + (c) * (offset)[2] + (d)))
#define A(a, b, c, d) ACCESS(in, a, b, c, d, in_offset)
#define B(a, b, c, d) ACCESS(weight, a, b, c, d, weight_offset)
#define C(a, b, c, d) ACCESS(out, a, b, c, d, out_offset)
    int n = int(threadIdx.x + blockIdx.x * blockDim.x) / out_channels;
    int c = int(threadIdx.x + blockIdx.x * blockDim.x) % out_channels;
    int h = int(threadIdx.y + blockIdx.y * blockDim.y);
    int w = int(threadIdx.z + blockIdx.z * blockDim.z);
    if(n < batch_size && c < out_channels && h < out_height && w < out_width) {
        float acc = 0.0;
        int in_h = h * stride_h - padding_h;
        int in_w = w * stride_w - padding_w;
        for(int in_c = 0; in_c < in_channels; in_c++) {
            for(int hh = 0; hh < kernel_h; hh++) {
                for(int ww = 0; ww < kernel_w; ww++) {
                    // acc += in[n, in_c, in_h + hh, in_w + ww] * weight[c, in_c, hh, ww];
                    acc += A(n, in_c, in_h + hh, in_w + ww) * B(c, in_c, hh, ww);
                }
            }
        }
        // out[n, c, h, w] = acc;
        C(n, c, h, w) = acc;
    }
#undef ACCESS
}

DLL void Conv2dReference(int num_args, const int *arg_types, void **args) {
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

    int out_h = (height + 2 * padding_h - kernel_h) / stride_h + 1;
    int out_w = (width + 2 * padding_w - kernel_w) / stride_w + 1;
    dim3 block(16, 4, 4);
    dim3 grid((batch_size * out_channels + block.x - 1) / block.x,
              (out_h + block.y - 1) / block.y, (out_w + block.z - 1) / block.z);
    conv2d_ref_kernel<<<grid, block>>>(batch_size, in_channels, height, width, out_channels, kernel_h,
                                       kernel_w, padding_h, padding_w, stride_h, stride_w, in, weight, out);
}
