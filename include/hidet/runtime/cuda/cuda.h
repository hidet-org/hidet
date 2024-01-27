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
#include <hidet/runtime/common.h>

typedef enum {
    cudaMemcpyHostToHost = 0,
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3,
    cudaMemcpyDefault = 4
} cudaMemcpyKind;

DLL int hidet_cuda_device_count();
DLL int hidet_cuda_get_device();
DLL void hidet_cuda_set_device(int device);
DLL void hidet_cuda_malloc(void **devPtr, size_t size);
DLL void hidet_cuda_free(void *devPtr);
DLL void hidet_cuda_memcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind);

