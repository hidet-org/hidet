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
#pragma once
#define HIDET_CUDNN_MAX_GPUS 32

#include <hidet/runtime/common.h>

struct cudnnContext;
typedef struct cudnnContext *cudnnHandle_t;

typedef void *cudnnBackendDescriptor_t;

/* Legacy API */
struct cudnnTensorStruct;
struct cudnnFilterStruct;
struct cudnnConvolutionStruct;

typedef struct cudnnTensorStruct *cudnnTensorDescriptor_t;
typedef struct cudnnFilterStruct *cudnnFilterDescriptor_t;
typedef struct cudnnConvolutionStruct *cudnnConvolutionDescriptor_t;

struct CudnnContext {
    cudnnHandle_t handles[HIDET_CUDNN_MAX_GPUS];
    static CudnnContext *global();
    static cudnnHandle_t current_handle();
};

DLL void hidet_cudnn_set_library_path(const char *path);
