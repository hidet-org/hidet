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
#include <hidet/runtime/cuda/context.h>
#include <hidet/runtime/cuda/cublas.h>
#include <hidet/runtime/cuda/cuda.h>
#include <hidet/runtime/logging.h>
// #include <cublas_v2.h>
// #include <cublas_v2.h>
#include "./utils.h"

// types defined in <cublas_v2.h>,
// copyright (c)
// NVIDIA Corporation. All rights reserved.
typedef enum {
    CUBLAS_STATUS_SUCCESS = 0,
    CUBLAS_STATUS_NOT_INITIALIZED = 1,
    CUBLAS_STATUS_ALLOC_FAILED = 3,
    CUBLAS_STATUS_INVALID_VALUE = 7,
    CUBLAS_STATUS_ARCH_MISMATCH = 8,
    CUBLAS_STATUS_MAPPING_ERROR = 11,
    CUBLAS_STATUS_EXECUTION_FAILED = 13,
    CUBLAS_STATUS_INTERNAL_ERROR = 14,
    CUBLAS_STATUS_NOT_SUPPORTED = 15,
    CUBLAS_STATUS_LICENSE_ERROR = 16
} cublasStatus_t;

typedef enum {
    CUBLAS_OP_N = 0,
    CUBLAS_OP_T = 1,
    CUBLAS_OP_C = 2,
} cublasOperation_t;

typedef enum cudaDataType_t {
    CUDA_R_16F = 2,
    CUDA_C_16F = 6,
    CUDA_R_16BF = 14,
    CUDA_C_16BF = 15,
    CUDA_R_32F = 0,
    CUDA_C_32F = 4,
    CUDA_R_64F = 1,
    CUDA_C_64F = 5,
    CUDA_R_4I = 16,
    CUDA_C_4I = 17,
    CUDA_R_4U = 18,
    CUDA_C_4U = 19,
    CUDA_R_8I = 3,
    CUDA_C_8I = 7,
    CUDA_R_8U = 8,
    CUDA_C_8U = 9,
    CUDA_R_16I = 20,
    CUDA_C_16I = 21,
    CUDA_R_16U = 22,
    CUDA_C_16U = 23,
    CUDA_R_32I = 10,
    CUDA_C_32I = 11,
    CUDA_R_32U = 12,
    CUDA_C_32U = 13,
    CUDA_R_64I = 24,
    CUDA_C_64I = 25,
    CUDA_R_64U = 26,
    CUDA_C_64U = 27,
    CUDA_R_8F_E4M3 = 28, /* real as a nv_fp8_e4m3 */
    CUDA_R_8F_E5M2 = 29, /* real as a nv_fp8_e5m2 */
} cudaDataType;

/* Enum for compute type
 *
 * - default types provide best available performance using all available hardware features
 *   and guarantee internal storage precision with at least the same precision and range;
 * - _PEDANTIC types ensure standard arithmetic and exact specified internal storage format;
 * - _FAST types allow for some loss of precision to enable higher throughput arithmetic.
 */
typedef enum {
    CUBLAS_COMPUTE_16F = 64,           /* half - default */
    CUBLAS_COMPUTE_16F_PEDANTIC = 65,  /* half - pedantic */
    CUBLAS_COMPUTE_32F = 68,           /* float - default */
    CUBLAS_COMPUTE_32F_PEDANTIC = 69,  /* float - pedantic */
    CUBLAS_COMPUTE_32F_FAST_16F = 74,  /* float - fast, allows down-converting inputs to half or TF32 */
    CUBLAS_COMPUTE_32F_FAST_16BF = 75, /* float - fast, allows down-converting inputs to bfloat16 or TF32 */
    CUBLAS_COMPUTE_32F_FAST_TF32 = 77, /* float - fast, allows down-converting inputs to TF32 */
    CUBLAS_COMPUTE_64F = 70,           /* double - default */
    CUBLAS_COMPUTE_64F_PEDANTIC = 71,  /* double - pedantic */
    CUBLAS_COMPUTE_32I = 72,           /* signed 32-bit int - default */
    CUBLAS_COMPUTE_32I_PEDANTIC = 73,  /* signed 32-bit int - pedantic */
} cublasComputeType_t;

typedef enum {
    CUBLAS_GEMM_DEFAULT = -1,
} cublasGemmAlgo_t;

// define cublas api functions
typedef const char *(*cublasGetStatusName_t)(cublasStatus_t status);
typedef const char *(*cublasGetStatusString_t)(cublasStatus_t status);
typedef cublasStatus_t (*cublasCreate_t)(cublasHandle_t *handle);
typedef cublasStatus_t (*cublasSetStream_t)(cublasHandle_t handle, cudaStream_t streamId);
typedef cublasStatus_t (*cublasGemmEx_t)(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
                                         int m, int n, int k, const void *alpha, const void *A, cudaDataType_t Atype,
                                         int lda, const void *B, cudaDataType_t Btype, int ldb, const void *beta,
                                         void *C, cudaDataType_t Ctype, int ldc, cublasComputeType_t computeType,
                                         cublasGemmAlgo_t algo);
typedef cublasStatus_t (*cublasGemmStridedBatchedEx_t)(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const void *alpha,
    const void *A, cudaDataType_t Atype, int lda, long long int strideA, const void *B, cudaDataType_t Btype, int ldb,
    long long int strideB, const void *beta, void *C, cudaDataType_t Ctype, int ldc, long long int strideC,
    int batchCount, cublasComputeType_t computeType, cublasGemmAlgo_t algo);
typedef cublasStatus_t (*cublasGemmBatchedEx_t)(cublasHandle_t handle, cublasOperation_t transa,
                                                cublasOperation_t transb, int m, int n, int k, const void *alpha,
                                                const void *const Aarray[], cudaDataType_t Atype, int lda,
                                                const void *const Barray[], cudaDataType_t Btype, int ldb,
                                                const void *beta, void *const Carray[], cudaDataType_t Ctype, int ldc,
                                                int batchCount, cublasComputeType_t computeType, cublasGemmAlgo_t algo);

// cublas api functions
static cublasCreate_t cublasCreate;
static cublasSetStream_t cublasSetStream;
static cublasGetStatusName_t cublasGetStatusName;
static cublasGetStatusString_t cublasGetStatusString;
static cublasGemmEx_t cublasGemmEx;
static cublasGemmStridedBatchedEx_t cublasGemmStridedBatchedEx;
static cublasGemmBatchedEx_t cublasGemmBatchedEx;

static std::string library_path;
static void *libcublas = nullptr;

// utility functions
#define CHECK_CUBLAS(status)                                                                                         \
    do {                                                                                                             \
        cublasStatus_t err = (status);                                                                               \
        if (err != 0) {                                                                                              \
            LOG(FATAL) << "cuBLAS error: " << cublasGetStatusString(err) << " (" << cublasGetStatusName(err) << ")"; \
        }                                                                                                            \
    } while (0)

static void set_alpha_beta(const void **p_alpha, const void **p_beta, cublasComputeType_t c, cudaDataType_t tc) {
    if (tc == CUDA_C_32F || tc == CUDA_C_64F) {
        LOG(FATAL) << "NotImplementedError: complex numbers are not supported yet" << std::endl;
    }

    if (c == CUBLAS_COMPUTE_16F || c == CUBLAS_COMPUTE_16F_PEDANTIC) {
        static const int16_t alpha = 0x3c00;  // half(1.0)
        static const int16_t beta = 0x0000;   // half(0.0)
        *p_alpha = &alpha;
        *p_beta = &beta;
    } else if (c == CUBLAS_COMPUTE_32F || c == CUBLAS_COMPUTE_32F_PEDANTIC || c == CUBLAS_COMPUTE_32F_FAST_16F ||
               c == CUBLAS_COMPUTE_32F_FAST_16BF || c == CUBLAS_COMPUTE_32F_FAST_TF32) {
        static const float alpha = 1.0f;
        static const float beta = 0.0f;
        *p_alpha = &alpha;
        *p_beta = &beta;
    } else if (c == CUBLAS_COMPUTE_32I || c == CUBLAS_COMPUTE_32I_PEDANTIC) {
        static const int32_t alpha = 1;
        static const int32_t beta = 0;
        *p_alpha = &alpha;
        *p_beta = &beta;
    } else if (c == CUBLAS_COMPUTE_64F || c == CUBLAS_COMPUTE_64F_PEDANTIC) {
        static const double alpha = 1.0;
        static const double beta = 0.0;
        *p_alpha = &alpha;
        *p_beta = &beta;
    } else {
        LOG(FATAL) << "Unsupported compute type: " << c;
    }
}

static void lazy_load_cublas() {
    if (libcublas == nullptr) {
        // load cublas shared library
        const char *libpath;
        if (library_path.empty()) {
            libpath = "libcublas.so";
        } else {
            libpath = library_path.c_str();
        }
        libcublas = dlopen(libpath, RTLD_LAZY);
        if (libcublas == nullptr) {
            LOG(FATAL) << "Failed to load cublas library: " << libpath << dlerror();
        }

        // load api functions
        cublasCreate = get_symbol<cublasCreate_t>(libcublas, "cublasCreate_v2");
        cublasSetStream = get_symbol<cublasSetStream_t>(libcublas, "cublasSetStream_v2");
        cublasGetStatusName = get_symbol<cublasGetStatusName_t>(libcublas, "cublasGetStatusName");
        cublasGetStatusString = get_symbol<cublasGetStatusString_t>(libcublas, "cublasGetStatusString");
        cublasGemmEx = get_symbol<cublasGemmEx_t>(libcublas, "cublasGemmEx");
        cublasGemmStridedBatchedEx = get_symbol<cublasGemmStridedBatchedEx_t>(libcublas, "cublasGemmStridedBatchedEx");
        cublasGemmBatchedEx = get_symbol<cublasGemmBatchedEx_t>(libcublas, "cublasGemmBatchedEx");
    }
}

CublasContext *CublasContext::global() {
    static CublasContext instance;
    static bool initialized = false;

    if (!initialized) {
        // create cublas handle for each gpu
        int count = hidet_cuda_device_count();
        assert(count <= HIDET_CUBLAS_MAX_GPUS);

        int current_device = hidet_cuda_get_device();
        for (int i = 0; i < count; i++) {
            hidet_cuda_set_device(i);
            CHECK_CUBLAS(cublasCreate(&instance.handles[i]));
        }
        hidet_cuda_set_device(current_device);

        initialized = true;
    }
    return &instance;
}

cublasHandle_t CublasContext::current_handle() {
    return CublasContext::global()->handles[hidet_cuda_get_device()];
}

// hidet cublas api functions
DLL void hidet_cublas_set_library_path(const char *path) {
    if (path) {
        library_path = path;
    }
}

DLL void hidet_cublas_gemm(int m, int n, int k, int ta, int tb, int tc, void *ptr_a, void *ptr_b, void *ptr_c,
                           bool trans_a, bool trans_b, int compute_type) {
    try {
        lazy_load_cublas();

        // Set the stream to the current stream
        cudaStream_t cur_stream = get_cuda_stream();
        CHECK_CUBLAS(cublasSetStream(CublasContext::current_handle(), cur_stream));

        const void *p_alpha = nullptr;
        const void *p_beta = nullptr;

        set_alpha_beta(&p_alpha, &p_beta, cublasComputeType_t(compute_type), cudaDataType_t(tc));

        // we apply c^T = b^T @ a^T (c = a @ b) here
        CHECK_CUBLAS(cublasGemmEx(CublasContext::current_handle(),
                                  trans_a ? cublasOperation_t::CUBLAS_OP_T : cublasOperation_t::CUBLAS_OP_N,
                                  trans_b ? cublasOperation_t::CUBLAS_OP_T : cublasOperation_t::CUBLAS_OP_N, n, m, k,
                                  p_alpha, ptr_b, cudaDataType(tb),
                                  n,  // ldb
                                  ptr_a, cudaDataType(ta),
                                  k,  // lda
                                  p_beta, ptr_c, cudaDataType(tc),
                                  n,  // ldc
                                  cublasComputeType_t(compute_type), cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT));
    } catch (HidetException &e) {
        hidet_set_last_error(e.what());
        return;
    }
}

DLL void hidet_cublas_strided_gemm(int b, int m, int n, int k, int ta, int tb, int tc, void *ptr_a, void *ptr_b,
                                   void *ptr_c, int64_t sa, int64_t sb, int64_t sc, bool trans_a, bool trans_b,
                                   int compute_type) {
    try {
        lazy_load_cublas();

        // Set the stream to the current stream
        cudaStream_t cur_stream = get_cuda_stream();
        CHECK_CUBLAS(cublasSetStream(CublasContext::current_handle(), cur_stream));

        const void *p_alpha = nullptr;
        const void *p_beta = nullptr;

        set_alpha_beta(&p_alpha, &p_beta, cublasComputeType_t(compute_type), cudaDataType_t(tc));

        CHECK_CUBLAS(cublasGemmStridedBatchedEx(
            CublasContext::current_handle(), trans_a ? cublasOperation_t::CUBLAS_OP_T : cublasOperation_t::CUBLAS_OP_N,
            trans_b ? cublasOperation_t::CUBLAS_OP_T : cublasOperation_t::CUBLAS_OP_N, n, m, k, p_alpha,
            // b^t
            ptr_b, cudaDataType(tb),
            n,   // ldb
            sb,  // strideB
            // a^t
            ptr_a, cudaDataType(ta),
            k,   // lda
            sa,  // strideA
            p_beta,
            // c^t
            ptr_c, cudaDataType(tc),
            n,   // ldc
            sc,  // strideC
            b,   // batchCount
            cublasComputeType_t(compute_type), cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT));
    } catch (HidetException &e) {
        hidet_set_last_error(e.what());
        return;
    }
}

DLL void hidet_cublas_batched_gemm(int b, int m, int n, int k, int ta, int tb, int tc, void **ptr_a, void **ptr_b,
                                   void **ptr_c, bool trans_a, bool trans_b, int compute_type) {
    try {
        lazy_load_cublas();

        // Set the stream to the current stream
        cudaStream_t cur_stream = get_cuda_stream();
        CHECK_CUBLAS(cublasSetStream(CublasContext::current_handle(), cur_stream));

        const void *p_alpha = nullptr;
        const void *p_beta = nullptr;

        set_alpha_beta(&p_alpha, &p_beta, cublasComputeType_t(compute_type), cudaDataType_t(tc));

        static void **ptr_a_device, **ptr_b_device, **ptr_c_device;
        static int
            cur_device_ptr_size;  // Size of device memory currently allocated for each of the three a,b,c arrays.

        // Allocate device memory
        // first use synchronous versions of malloc and memcpy, later switch to async versions
        if (b > cur_device_ptr_size) {
            if (cur_device_ptr_size > 0) {
                hidet_cuda_free_async((void *)ptr_a_device, cur_stream);
                hidet_cuda_free_async((void *)ptr_b_device, cur_stream);
                hidet_cuda_free_async((void *)ptr_c_device, cur_stream);
            }
            ptr_a_device = (void **)hidet_cuda_malloc_async(b * sizeof(void *), cur_stream);
            ptr_b_device = (void **)hidet_cuda_malloc_async(b * sizeof(void *), cur_stream);
            ptr_c_device = (void **)hidet_cuda_malloc_async(b * sizeof(void *), cur_stream);

            cur_device_ptr_size = b;
        }

        // Copy input arrays (A and B) from host to device
        hidet_cuda_memcpy_async((void *)ptr_a_device, (void *)ptr_a, b * sizeof(void *), cudaMemcpyHostToDevice,
                                cur_stream);
        hidet_cuda_memcpy_async((void *)ptr_b_device, (void *)ptr_b, b * sizeof(void *), cudaMemcpyHostToDevice,
                                cur_stream);
        hidet_cuda_memcpy_async((void *)ptr_c_device, (void *)ptr_c, b * sizeof(void *), cudaMemcpyHostToDevice,
                                cur_stream);

        CHECK_CUBLAS(cublasGemmBatchedEx(
            CublasContext::current_handle(), trans_a ? cublasOperation_t::CUBLAS_OP_T : cublasOperation_t::CUBLAS_OP_N,
            trans_b ? cublasOperation_t::CUBLAS_OP_T : cublasOperation_t::CUBLAS_OP_N, n, m, k, p_alpha,
            // b^t
            ptr_b_device, cudaDataType(tb),
            n,  // ldb
            // a^t
            ptr_a_device, cudaDataType(ta),
            k,  // lda
            p_beta,
            // c^t
            ptr_c_device, cudaDataType(tc),
            n,  // ldc
            b,  // batchCount
            cublasComputeType_t(compute_type), cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT));
    } catch (HidetException &e) {
        hidet_set_last_error(e.what());
        return;
    }
}
