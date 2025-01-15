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
typedef cudaError_t (*cudaGetDeviceCount_t)(int *count);
typedef cudaError_t (*cudaGetDevice_t)(int *device);
typedef cudaError_t (*cudaSetDevice_t)(int device);
typedef cudaError_t (*cudaMalloc_t)(void **devPtr, size_t size);
typedef cudaError_t (*cudaMallocAsync_t)(void **devPtr, size_t size, cudaStream_t stream);
typedef cudaError_t (*cudaFree_t)(void *devPtr);
typedef cudaError_t (*cudaFreeAsync_t)(void *devPtr, cudaStream_t stream);
typedef cudaError_t (*cudaMemcpy_t)(void *dst, const void *src, size_t count, cudaMemcpyKind kind);
typedef cudaError_t (*cudaMemcpyAsync_t)(void *dst, const void *src, size_t count, cudaMemcpyKind kind,
                                         cudaStream_t stream);
typedef const char *(*cudaGetErrorString_t)(cudaError_t error);

static std::string library_path;
static void *libcudart = nullptr;
static cudaGetDeviceCount_t cudaGetDeviceCount = nullptr;
static cudaGetDevice_t cudaGetDevice = nullptr;
static cudaSetDevice_t cudaSetDevice = nullptr;
static cudaMalloc_t cudaMalloc = nullptr;
static cudaMallocAsync_t cudaMallocAsync = nullptr;
static cudaFree_t cudaFree = nullptr;
static cudaFreeAsync_t cudaFreeAsync = nullptr;
static cudaMemcpy_t cudaMemcpy = nullptr;
static cudaMemcpyAsync_t cudaMemcpyAsync = nullptr;
static cudaGetErrorString_t cudaGetErrorString = nullptr;

// load cuda runtime APIs
static inline void lazy_load_cuda_runtime() {
    if (libcudart == nullptr) {
        const char *libpath;
        if (library_path.empty()) {
            libpath = "libcudart.so";
        } else {
            libpath = library_path.c_str();
        }
        libcudart = dlopen(libpath, RTLD_LAZY);

        if (libcudart == nullptr) {
            LOG(FATAL) << "Failed to load libcudart.so: " << dlerror();
        }

        cudaGetDeviceCount = get_symbol<cudaGetDeviceCount_t>(libcudart, "cudaGetDeviceCount");
        cudaGetDevice = get_symbol<cudaGetDevice_t>(libcudart, "cudaGetDevice");
        cudaSetDevice = get_symbol<cudaSetDevice_t>(libcudart, "cudaSetDevice");
        cudaMalloc = get_symbol<cudaMalloc_t>(libcudart, "cudaMalloc");
        cudaMallocAsync = get_symbol<cudaMallocAsync_t>(libcudart, "cudaMallocAsync");
        cudaFree = get_symbol<cudaFree_t>(libcudart, "cudaFree");
        cudaFreeAsync = get_symbol<cudaFreeAsync_t>(libcudart, "cudaFreeAsync");
        cudaMemcpy = get_symbol<cudaMemcpy_t>(libcudart, "cudaMemcpy");
        cudaMemcpyAsync = get_symbol<cudaMemcpyAsync_t>(libcudart, "cudaMemcpyAsync");
        cudaGetErrorString = get_symbol<cudaGetErrorString_t>(libcudart, "cudaGetErrorString");
    }
}

#define CHECK_CUDA(status)                                           \
    do {                                                             \
        cudaError_t err = (status);                                  \
        if (err != 0) {                                              \
            LOG(FATAL) << "CUDA error: " << cudaGetErrorString(err); \
        }                                                            \
    } while (0)

// Hidet exported APIs
DLL void hidet_cuda_set_library_path(const char *path) {
    try {
        library_path = path;
    } catch (HidetException &e) {
        hidet_set_last_error(e.what());
        return;
    }
}

DLL int hidet_cuda_device_count() {
    try {
        lazy_load_cuda_runtime();
        int count = 0;
        CHECK_CUDA(cudaGetDeviceCount(&count));
        return count;
    } catch (HidetException &e) {
        hidet_set_last_error(e.what());
        return 0;
    }
}

DLL int hidet_cuda_get_device() {
    try {
        lazy_load_cuda_runtime();
        int current_device = -1;
        CHECK_CUDA(cudaGetDevice(&current_device));
        return current_device;
    } catch (HidetException &e) {
        hidet_set_last_error(e.what());
        return -1;
    }
}

DLL void hidet_cuda_set_device(int device) {
    try {
        lazy_load_cuda_runtime();
        CHECK_CUDA(cudaSetDevice(device));

    } catch (HidetException &e) {
        hidet_set_last_error(e.what());
        return;
    }
}
DLL void *hidet_cuda_malloc(size_t size) {
    try {
        lazy_load_cuda_runtime();
        void *devPtr;
        CHECK_CUDA(cudaMalloc(&devPtr, size));
        return devPtr;
    } catch (HidetException &e) {
        hidet_set_last_error(e.what());
        return nullptr;
    }
}

DLL void *hidet_cuda_malloc_async(size_t size, cudaStream_t stream) {
    try {
        lazy_load_cuda_runtime();
        void *devPtr;
        CHECK_CUDA(cudaMallocAsync(&devPtr, size, stream));
        return devPtr;
    } catch (HidetException &e) {
        hidet_set_last_error(e.what());
        return nullptr;
    }
}

DLL void hidet_cuda_free(void *devPtr) {
    try {
        lazy_load_cuda_runtime();
        CHECK_CUDA(cudaFree(devPtr));
    } catch (HidetException &e) {
        hidet_set_last_error(e.what());
        return;
    }
}

DLL void hidet_cuda_free_async(void *devPtr, cudaStream_t stream) {
    try {
        lazy_load_cuda_runtime();
        CHECK_CUDA(cudaFreeAsync(devPtr, stream));
    } catch (HidetException &e) {
        hidet_set_last_error(e.what());
        return;
    }
}

DLL void hidet_cuda_memcpy(void *dst, const void *src, size_t count, cudaMemcpyKind kind) {
    try {
        lazy_load_cuda_runtime();
        CHECK_CUDA(cudaMemcpy(dst, src, count, kind));
    } catch (HidetException &e) {
        hidet_set_last_error(e.what());
        return;
    }
}

DLL void hidet_cuda_memcpy_async(void *dst, const void *src, size_t count, cudaMemcpyKind kind, cudaStream_t stream) {
    try {
        lazy_load_cuda_runtime();
        CHECK_CUDA(cudaMemcpyAsync(dst, src, count, kind, stream));
    } catch (HidetException &e) {
        hidet_set_last_error(e.what());
        return;
    }
}