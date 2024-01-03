// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include <dlfcn.h>
#include <hidet/runtime/cuda/cuda.h>
#include "./utils.h"

// CUDA runtime APIs
typedef int cudaError_t;
typedef cudaError_t (*cudaGetDeviceCount_t)(int* count);
typedef cudaError_t (*cudaGetDevice_t)(int* device);
typedef cudaError_t (*cudaSetDevice_t)(int device);
typedef const char* (*cudaGetErrorString_t)(cudaError_t error);

static std::string library_path;
static void* libcudart = nullptr;
static cudaGetDeviceCount_t cudaGetDeviceCount = nullptr;
static cudaGetDevice_t cudaGetDevice = nullptr;
static cudaSetDevice_t cudaSetDevice = nullptr;
static cudaGetErrorString_t cudaGetErrorString = nullptr;

// load cuda runtime APIs
static inline void lazy_load_cuda_runtime() {
    if(libcudart == nullptr) {
        const char* libpath;
        if(library_path.empty()) {
            libpath = "libcudart.so";
        } else {
            libpath = library_path.c_str();
        }
        libcudart = dlopen(libpath, RTLD_LAZY);

        if(libcudart == nullptr) {
            LOG(FATAL) << "Failed to load libcudart.so: " << dlerror();
        }

        cudaGetDeviceCount = get_symbol<cudaGetDeviceCount_t>(libcudart, "cudaGetDeviceCount");
        cudaGetDevice = get_symbol<cudaGetDevice_t>(libcudart, "cudaGetDevice");
        cudaSetDevice = get_symbol<cudaSetDevice_t>(libcudart, "cudaSetDevice");
        cudaGetErrorString = get_symbol<cudaGetErrorString_t>(libcudart, "cudaGetErrorString");
    }
}

#define CHECK_CUDA(status) do{                                      \
    cudaError_t err = (status);                                     \
    if (err != 0) {                                                 \
        LOG(FATAL) << "CUDA error: " << cudaGetErrorString(err);    \
    }                                                               \
} while(0)

// Hidet exported APIs
DLL void hidet_cuda_set_library_path(const char* path) {
    library_path = path;
}

DLL int hidet_cuda_device_count() {
    lazy_load_cuda_runtime();
    int count = 0;
    CHECK_CUDA(cudaGetDeviceCount(&count));
    return count;
}

DLL int hidet_cuda_get_device() {
    lazy_load_cuda_runtime();
    int current_device = -1;
    CHECK_CUDA(cudaGetDevice(&current_device));
    return current_device;
}

DLL void hidet_cuda_set_device(int device) {
    lazy_load_cuda_runtime();
    CHECK_CUDA(cudaSetDevice(device));
}
