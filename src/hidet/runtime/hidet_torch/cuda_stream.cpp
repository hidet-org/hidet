#include <c10/cuda/CUDAStream.h>
#include <hidet/runtime/common.h>

DLL void *hidet_get_current_torch_cuda_stream() {
    return at::cuda::getCurrentCUDAStream().stream();
}
