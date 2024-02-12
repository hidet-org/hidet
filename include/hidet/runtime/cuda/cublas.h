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
//#include <cublas_v2.h>

#define HIDET_CUBLAS_MAX_GPUS 32

typedef void* cublasHandle_t;

struct CublasContext {
    cublasHandle_t handles[HIDET_CUBLAS_MAX_GPUS];   // cublas handle for each gpu on this node
    static CublasContext* global();
    static cublasHandle_t current_handle();
};

DLL void hidet_cublas_set_library_path(const char* path);

// kernel functions
DLL void hidet_cublas_gemm(
    int m, int n, int k, int ta, int tb, int tc, void *ptr_a, void* ptr_b, void* ptr_c, bool trans_a, bool trans_b,
    int compute_type
);

DLL void hidet_cublas_strided_gemm(
    int b, int m, int n, int k, int ta, int tb, int tc, void *ptr_a, void* ptr_b, void* ptr_c,
    int64_t sa, int64_t sb, int64_t sc,
    bool trans_a, bool trans_b, int compute_type
);

DLL void hidet_cublas_batched_gemm(
    int b, int m, int n, int k, int ta, int tb, int tc, void **ptr_a, void **ptr_b, void **ptr_c,
    bool trans_a, bool trans_b, int compute_type
);
