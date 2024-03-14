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

#include <hidet/runtime/cuda/cuda.h>
#include <hidet/runtime/cuda/context.h>
#include <hidet/runtime/cuda/cudnn.h>
#include <hidet/runtime/logging.h>
#include "./utils.h"

/*
 * CUDNN return codes - defined in cudnn_ops_infer_v8.h
 */
typedef enum {
    CUDNN_STATUS_SUCCESS                      = 0,
    CUDNN_STATUS_NOT_INITIALIZED              = 1,
    CUDNN_STATUS_ALLOC_FAILED                 = 2,
    CUDNN_STATUS_BAD_PARAM                    = 3,
    CUDNN_STATUS_INTERNAL_ERROR               = 4,
    CUDNN_STATUS_INVALID_VALUE                = 5,
    CUDNN_STATUS_ARCH_MISMATCH                = 6,
    CUDNN_STATUS_MAPPING_ERROR                = 7,
    CUDNN_STATUS_EXECUTION_FAILED             = 8,
    CUDNN_STATUS_NOT_SUPPORTED                = 9,
    CUDNN_STATUS_LICENSE_ERROR                = 10,
    CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING = 11,
    CUDNN_STATUS_RUNTIME_IN_PROGRESS          = 12,
    CUDNN_STATUS_RUNTIME_FP_OVERFLOW          = 13,
    CUDNN_STATUS_VERSION_MISMATCH             = 14,
} cudnnStatus_t;

/*
* CUDNN Descriptor Types - defined in cudnn_backend_v8.h
*/
typedef enum {
    CUDNN_BACKEND_POINTWISE_DESCRIPTOR = 0,
    CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR,
    CUDNN_BACKEND_ENGINE_DESCRIPTOR,
    CUDNN_BACKEND_ENGINECFG_DESCRIPTOR,
    CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR,
    CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR,
    CUDNN_BACKEND_INTERMEDIATE_INFO_DESCRIPTOR,
    CUDNN_BACKEND_KNOB_CHOICE_DESCRIPTOR,
    CUDNN_BACKEND_KNOB_INFO_DESCRIPTOR,
    CUDNN_BACKEND_LAYOUT_INFO_DESCRIPTOR,
    CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR,
    CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR,
    CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_DATA_DESCRIPTOR,
    CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR,
    CUDNN_BACKEND_OPERATION_GEN_STATS_DESCRIPTOR,
    CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR,
    CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR,
    CUDNN_BACKEND_TENSOR_DESCRIPTOR,
    CUDNN_BACKEND_MATMUL_DESCRIPTOR,
    CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR,
    CUDNN_BACKEND_OPERATION_BN_FINALIZE_STATISTICS_DESCRIPTOR,
    CUDNN_BACKEND_REDUCTION_DESCRIPTOR,
    CUDNN_BACKEND_OPERATION_REDUCTION_DESCRIPTOR,
    CUDNN_BACKEND_OPERATION_BN_BWD_WEIGHTS_DESCRIPTOR,
} cudnnBackendDescriptorType_t;

/*
 * CUDNN data type - defined in cudnn_ops_infer_v8.h
 */
typedef enum {
    CUDNN_DATA_FLOAT    = 0,
    CUDNN_DATA_DOUBLE   = 1,
    CUDNN_DATA_HALF     = 2,
    CUDNN_DATA_INT8     = 3,
    CUDNN_DATA_INT32    = 4,
    CUDNN_DATA_INT8x4   = 5,
    CUDNN_DATA_UINT8    = 6,
    CUDNN_DATA_UINT8x4  = 7,
    CUDNN_DATA_INT8x32  = 8,
    CUDNN_DATA_BFLOAT16 = 9,
    CUDNN_DATA_INT64    = 10,
} cudnnDataType_t;

/*
* CUDNN Backend Attribute Names - defined in cudnn_backend_v8.h
*/
typedef enum {
    CUDNN_ATTR_POINTWISE_MODE                  = 0,
    CUDNN_ATTR_POINTWISE_MATH_PREC             = 1,
    CUDNN_ATTR_POINTWISE_NAN_PROPAGATION       = 2,
    CUDNN_ATTR_POINTWISE_RELU_LOWER_CLIP       = 3,
    CUDNN_ATTR_POINTWISE_RELU_UPPER_CLIP       = 4,
    CUDNN_ATTR_POINTWISE_RELU_LOWER_CLIP_SLOPE = 5,
    CUDNN_ATTR_POINTWISE_ELU_ALPHA             = 6,
    CUDNN_ATTR_POINTWISE_SOFTPLUS_BETA         = 7,
    CUDNN_ATTR_POINTWISE_SWISH_BETA            = 8,

    CUDNN_ATTR_CONVOLUTION_COMP_TYPE      = 100,
    CUDNN_ATTR_CONVOLUTION_CONV_MODE      = 101,
    CUDNN_ATTR_CONVOLUTION_DILATIONS      = 102,
    CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES = 103,
    CUDNN_ATTR_CONVOLUTION_POST_PADDINGS  = 104,
    CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS   = 105,
    CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS   = 106,

    CUDNN_ATTR_ENGINEHEUR_MODE            = 200,
    CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH = 201,
    CUDNN_ATTR_ENGINEHEUR_RESULTS         = 202,

    CUDNN_ATTR_ENGINECFG_ENGINE            = 300,
    CUDNN_ATTR_ENGINECFG_INTERMEDIATE_INFO = 301,
    CUDNN_ATTR_ENGINECFG_KNOB_CHOICES      = 302,

    CUDNN_ATTR_EXECUTION_PLAN_HANDLE                     = 400,
    CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG              = 401,
    CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE             = 402,
    CUDNN_ATTR_EXECUTION_PLAN_COMPUTED_INTERMEDIATE_UIDS = 403,
    CUDNN_ATTR_EXECUTION_PLAN_RUN_ONLY_INTERMEDIATE_UIDS = 404,

    CUDNN_ATTR_INTERMEDIATE_INFO_UNIQUE_ID            = 500,
    CUDNN_ATTR_INTERMEDIATE_INFO_SIZE                 = 501,
    CUDNN_ATTR_INTERMEDIATE_INFO_DEPENDENT_DATA_UIDS  = 502,
    CUDNN_ATTR_INTERMEDIATE_INFO_DEPENDENT_ATTRIBUTES = 503,

    CUDNN_ATTR_KNOB_CHOICE_KNOB_TYPE  = 600,
    CUDNN_ATTR_KNOB_CHOICE_KNOB_VALUE = 601,

    CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA        = 700,
    CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA         = 701,
    CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC    = 702,
    CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W            = 703,
    CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X            = 704,
    CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y            = 705,
    CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_ALPHA       = 706,
    CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_BETA        = 707,
    CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_CONV_DESC   = 708,
    CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_W           = 709,
    CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DX          = 710,
    CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_DATA_DY          = 711,
    CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_ALPHA     = 712,
    CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_BETA      = 713,
    CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_CONV_DESC = 714,
    CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DW        = 715,
    CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_X         = 716,
    CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DY        = 717,

    CUDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR = 750,
    CUDNN_ATTR_OPERATION_POINTWISE_XDESC         = 751,
    CUDNN_ATTR_OPERATION_POINTWISE_BDESC         = 752,
    CUDNN_ATTR_OPERATION_POINTWISE_YDESC         = 753,
    CUDNN_ATTR_OPERATION_POINTWISE_ALPHA1        = 754,
    CUDNN_ATTR_OPERATION_POINTWISE_ALPHA2        = 755,
    CUDNN_ATTR_OPERATION_POINTWISE_DXDESC        = 756,
    CUDNN_ATTR_OPERATION_POINTWISE_DYDESC        = 757,

    CUDNN_ATTR_OPERATION_GENSTATS_MODE      = 770,
    CUDNN_ATTR_OPERATION_GENSTATS_MATH_PREC = 771,
    CUDNN_ATTR_OPERATION_GENSTATS_XDESC     = 772,
    CUDNN_ATTR_OPERATION_GENSTATS_SUMDESC   = 773,
    CUDNN_ATTR_OPERATION_GENSTATS_SQSUMDESC = 774,

    CUDNN_ATTR_OPERATION_BN_FINALIZE_STATS_MODE                = 780,
    CUDNN_ATTR_OPERATION_BN_FINALIZE_MATH_PREC                 = 781,
    CUDNN_ATTR_OPERATION_BN_FINALIZE_Y_SUM_DESC                = 782,
    CUDNN_ATTR_OPERATION_BN_FINALIZE_Y_SQ_SUM_DESC             = 783,
    CUDNN_ATTR_OPERATION_BN_FINALIZE_SCALE_DESC                = 784,
    CUDNN_ATTR_OPERATION_BN_FINALIZE_BIAS_DESC                 = 785,
    CUDNN_ATTR_OPERATION_BN_FINALIZE_PREV_RUNNING_MEAN_DESC    = 786,
    CUDNN_ATTR_OPERATION_BN_FINALIZE_PREV_RUNNING_VAR_DESC     = 787,
    CUDNN_ATTR_OPERATION_BN_FINALIZE_UPDATED_RUNNING_MEAN_DESC = 788,
    CUDNN_ATTR_OPERATION_BN_FINALIZE_UPDATED_RUNNING_VAR_DESC  = 789,
    CUDNN_ATTR_OPERATION_BN_FINALIZE_SAVED_MEAN_DESC           = 790,
    CUDNN_ATTR_OPERATION_BN_FINALIZE_SAVED_INV_STD_DESC        = 791,
    CUDNN_ATTR_OPERATION_BN_FINALIZE_EQ_SCALE_DESC             = 792,
    CUDNN_ATTR_OPERATION_BN_FINALIZE_EQ_BIAS_DESC              = 793,
    CUDNN_ATTR_OPERATION_BN_FINALIZE_ACCUM_COUNT_DESC          = 794,
    CUDNN_ATTR_OPERATION_BN_FINALIZE_EPSILON_DESC              = 795,
    CUDNN_ATTR_OPERATION_BN_FINALIZE_EXP_AVERATE_FACTOR_DESC   = 796,

    CUDNN_ATTR_OPERATIONGRAPH_HANDLE              = 800,
    CUDNN_ATTR_OPERATIONGRAPH_OPS                 = 801,
    CUDNN_ATTR_OPERATIONGRAPH_ENGINE_GLOBAL_COUNT = 802,

    CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT       = 900,
    CUDNN_ATTR_TENSOR_DATA_TYPE            = 901,
    CUDNN_ATTR_TENSOR_DIMENSIONS           = 902,
    CUDNN_ATTR_TENSOR_STRIDES              = 903,
    CUDNN_ATTR_TENSOR_VECTOR_COUNT         = 904,
    CUDNN_ATTR_TENSOR_VECTORIZED_DIMENSION = 905,
    CUDNN_ATTR_TENSOR_UNIQUE_ID            = 906,
    CUDNN_ATTR_TENSOR_IS_VIRTUAL           = 907,
    CUDNN_ATTR_TENSOR_IS_BY_VALUE          = 908,

    CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS    = 1000,
    CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS = 1001,
    CUDNN_ATTR_VARIANT_PACK_INTERMEDIATES = 1002,
    CUDNN_ATTR_VARIANT_PACK_WORKSPACE     = 1003,

    CUDNN_ATTR_LAYOUT_INFO_TENSOR_UID = 1100,
    CUDNN_ATTR_LAYOUT_INFO_TYPES      = 1101,

    CUDNN_ATTR_KNOB_INFO_TYPE          = 1200,
    CUDNN_ATTR_KNOB_INFO_MAXIMUM_VALUE = 1201,
    CUDNN_ATTR_KNOB_INFO_MINIMUM_VALUE = 1202,
    CUDNN_ATTR_KNOB_INFO_STRIDE        = 1203,

    CUDNN_ATTR_ENGINE_OPERATION_GRAPH = 1300,
    CUDNN_ATTR_ENGINE_GLOBAL_INDEX    = 1301,
    CUDNN_ATTR_ENGINE_KNOB_INFO       = 1302,
    CUDNN_ATTR_ENGINE_NUMERICAL_NOTE  = 1303,
    CUDNN_ATTR_ENGINE_LAYOUT_INFO     = 1304,
    CUDNN_ATTR_ENGINE_BEHAVIOR_NOTE   = 1305,

    CUDNN_ATTR_MATMUL_COMP_TYPE = 1500,

    CUDNN_ATTR_OPERATION_MATMUL_ADESC                           = 1520,
    CUDNN_ATTR_OPERATION_MATMUL_BDESC                           = 1521,
    CUDNN_ATTR_OPERATION_MATMUL_CDESC                           = 1522,
    CUDNN_ATTR_OPERATION_MATMUL_DESC                            = 1523,
    CUDNN_ATTR_OPERATION_MATMUL_IRREGULARLY_STRIDED_BATCH_COUNT = 1524,

    CUDNN_ATTR_REDUCTION_OPERATOR  = 1600,
    CUDNN_ATTR_REDUCTION_COMP_TYPE = 1601,

    CUDNN_ATTR_OPERATION_REDUCTION_XDESC = 1610,
    CUDNN_ATTR_OPERATION_REDUCTION_YDESC = 1611,
    CUDNN_ATTR_OPERATION_REDUCTION_DESC  = 1612,

    CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_MATH_PREC        = 1620,
    CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_MEAN_DESC        = 1621,
    CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_INVSTD_DESC      = 1622,
    CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_BN_SCALE_DESC    = 1623,
    CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_X_DESC           = 1624,
    CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_DY_DESC          = 1625,
    CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_DBN_SCALE_DESC   = 1626,
    CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_DBN_BIAS_DESC    = 1627,
    CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_EQ_DY_SCALE_DESC = 1628,
    CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_EQ_X_SCALE_DESC  = 1629,
    CUDNN_ATTR_OPERATION_BN_BWD_WEIGHTS_EQ_BIAS          = 1630,
} cudnnBackendAttributeName_t;

/*
* CUDNN Backend Attribute Type - defined in cudnn_backend_v8.h
*/
typedef enum {
    CUDNN_TYPE_HANDLE = 0,
    CUDNN_TYPE_DATA_TYPE,
    CUDNN_TYPE_BOOLEAN,
    CUDNN_TYPE_INT64,
    CUDNN_TYPE_FLOAT,
    CUDNN_TYPE_DOUBLE,
    CUDNN_TYPE_VOID_PTR,
    CUDNN_TYPE_CONVOLUTION_MODE,
    CUDNN_TYPE_HEUR_MODE,
    CUDNN_TYPE_KNOB_TYPE,
    CUDNN_TYPE_NAN_PROPOGATION,
    CUDNN_TYPE_NUMERICAL_NOTE,
    CUDNN_TYPE_LAYOUT_TYPE,
    CUDNN_TYPE_ATTRIB_NAME,
    CUDNN_TYPE_POINTWISE_MODE,
    CUDNN_TYPE_BACKEND_DESCRIPTOR,
    CUDNN_TYPE_GENSTATS_MODE,
    CUDNN_TYPE_BN_FINALIZE_STATS_MODE,
    CUDNN_TYPE_REDUCTION_OPERATOR_TYPE,
    CUDNN_TYPE_BEHAVIOR_NOTE,
} cudnnBackendAttributeType_t;

/*
 *  convolution mode - defined in cudnn_cnn_infer_v8.h
 */
typedef enum { CUDNN_CONVOLUTION = 0, CUDNN_CROSS_CORRELATION = 1 } cudnnConvolutionMode_t;


// define cudnn Graph API functions
typedef cudnnStatus_t (*cudnnCreate_t)(cudnnHandle_t *handle);
typedef const char * (*cudnnGetErrorString_t)(cudnnStatus_t status);
typedef cudnnStatus_t (*cudnnSetStream_t)(
    cudnnHandle_t handle,
    cudaStream_t streamId);
typedef cudnnStatus_t (*cudnnBackendCreateDescriptor_t)(
    cudnnBackendDescriptorType_t descriptorType, 
    cudnnBackendDescriptor_t *descriptor);
typedef cudnnStatus_t (*cudnnBackendDestroyDescriptor_t)(cudnnBackendDescriptor_t descriptor);
typedef cudnnStatus_t (*cudnnBackendSetAttribute_t)(
    cudnnBackendDescriptor_t descriptor,
    cudnnBackendAttributeName_t attributeName,
    cudnnBackendAttributeType_t attributeType,
    int64_t elementCount,
    void *arrayOfElements);
typedef cudnnStatus_t (*cudnnBackendGetAttribute_t)(
    cudnnBackendDescriptor_t descriptor,
    cudnnBackendAttributeName_t attributeName,
    cudnnBackendAttributeType_t attributeType,
    int64_t requestedElementCount,
    int64_t *elementCount,
    void *arrayOfElements);
typedef cudnnStatus_t (*cudnnBackendFinalize_t)(cudnnBackendDescriptor_t descriptor);
typedef cudnnStatus_t (*cudnnBackendExecute_t)(
    cudnnHandle_t handle, cudnnBackendDescriptor_t executionPlan, cudnnBackendDescriptor_t varianPack);


// cudnn api functions
static cudnnCreate_t cudnnCreate;
static cudnnGetErrorString_t cudnnGetErrorString;
static cudnnSetStream_t cudnnSetStream;
static cudnnBackendCreateDescriptor_t cudnnBackendCreateDescriptor;
static cudnnBackendDestroyDescriptor_t cudnnBackendDestroyDescriptor;
static cudnnBackendSetAttribute_t cudnnBackendSetAttribute;
static cudnnBackendGetAttribute_t cudnnBackendGetAttribute;
static cudnnBackendFinalize_t cudnnBackendFinalize;
static cudnnBackendExecute_t cudnnBackendExecute;

static std::string library_path;
static void* libcudnn = nullptr;

// utility functions
#define CHECK_CUDNN(status) do {                                      \
    cudnnStatus_t err = (status);                                     \
    if(err != 0) {                                                    \
        LOG(FATAL) << "cuDNN error: " << cudnnGetErrorString(err);    \
    }                                                                 \
} while(0)

static cudnnBackendAttributeType_t get_attribute_type_from_compute_type(cudnnDataType_t computeType) {
    switch (computeType) {
        case CUDNN_DATA_FLOAT:
            return CUDNN_TYPE_FLOAT;
        case CUDNN_DATA_DOUBLE:
            return CUDNN_TYPE_DOUBLE;
        case CUDNN_DATA_INT64:
        case CUDNN_DATA_INT32:
            return CUDNN_TYPE_INT64;
        default:
            LOG(FATAL) << "Unsupported compute type: " << computeType;
            return CUDNN_TYPE_VOID_PTR;
    }
}

static void set_alpha_beta(void** p_alpha, void** p_beta, cudnnDataType_t c) {
    // There's no such thing as a cudnnComputeType_t type. As per the official example, the computeType is defined
    // in terms of cudnnDataType_t
    // cudnnBackendAttributeType_t only has support for FLOAT, DOUBLE, and INT64.
    if(c == CUDNN_DATA_FLOAT) {
        static float alpha = 1.0f;
        static float beta = 0.0f;
        *p_alpha = &alpha;
        *p_beta = &beta;
    } else if(c == CUDNN_DATA_DOUBLE) {
        static double alpha = 1.0;
        static double beta = 0.0;
        *p_alpha = &alpha;
        *p_beta = &beta;
    } else if(c == CUDNN_DATA_INT64 || c == CUDNN_DATA_INT32) {
        static int64_t alpha = 1;
        static int64_t beta = 0;
        *p_alpha = &alpha;
        *p_beta = &beta;
    } else {
        LOG(FATAL) << "Unsupported compute type: " << c;
    }
}

static void lazy_load_cudnn() {
    if(libcudnn == nullptr) {
        // load cudnn shared library
        const char* libpath;
        if(library_path.empty()) {
            libpath = "libcudnn.so";
        } else {
            libpath = library_path.c_str();
        }
        libcudnn = dlopen(libpath, RTLD_LAZY);
        if(libcudnn == nullptr) {
            LOG(FATAL) << "Failed to load cublas library: " << libpath << dlerror();
        }

        // load api functions
        cudnnCreate = get_symbol<cudnnCreate_t>(libcudnn, "cudnnCreate");
        cudnnGetErrorString = get_symbol<cudnnGetErrorString_t>(libcudnn, "cudnnGetErrorString");
        cudnnSetStream = get_symbol<cudnnSetStream_t>(libcudnn, "cudnnSetStream");
        cudnnBackendCreateDescriptor = get_symbol<cudnnBackendCreateDescriptor_t>(libcudnn, "cudnnBackendCreateDescriptor");
        cudnnBackendDestroyDescriptor = get_symbol<cudnnBackendDestroyDescriptor_t>(libcudnn, "cudnnBackendDestroyDescriptor");
        cudnnBackendSetAttribute = get_symbol<cudnnBackendSetAttribute_t>(libcudnn, "cudnnBackendSetAttribute");
        cudnnBackendGetAttribute = get_symbol<cudnnBackendGetAttribute_t>(libcudnn, "cudnnBackendGetAttribute");
        cudnnBackendFinalize = get_symbol<cudnnBackendFinalize_t>(libcudnn, "cudnnBackendFinalize");
        cudnnBackendExecute = get_symbol<cudnnBackendExecute_t>(libcudnn, "cudnnBackendExecute");
    }
}


CudnnContext* CudnnContext::global() {
    static CudnnContext instance;
    static bool initialized = false;

    if(!initialized) {
        // create cudnn handle for each gpu
        int count = hidet_cuda_device_count();
        assert(count <= HIDET_CUBLAS_MAX_GPUS);

        int current_device = hidet_cuda_get_device();
        for(int i = 0; i < count; i++) {
            hidet_cuda_set_device(i);
            CHECK_CUDNN(cudnnCreate(&instance.handles[i]));
        }
        hidet_cuda_set_device(current_device);

        initialized = true;
    }
    return &instance;
}

cudnnHandle_t CudnnContext::current_handle() {
    return CudnnContext::global()->handles[hidet_cuda_get_device()];
}


// hidet cudnn api functions
DLL void hidet_cudnn_set_library_path(const char* path) {
    if(path) {
        library_path = path;
    }
}

DLL void hidet_cudnn_conv2d(
    int n, int c, int h, int w, int k, int r, int s, int p, int q,
    void *ptr_x, void *ptr_w, void *ptr_y,
    int tx, int tw, int ty, int compute_type, 
    int pad_dim1, int pad_dim2, int str_dim1, int str_dim2, int dil_dim1, int dil_dim2
) {
    lazy_load_cudnn();
    
    cudnnHandle_t cur_handle = CudnnContext::current_handle();

    // Set the stream to the current stream
    cudaStream_t cur_stream = get_cuda_stream();
    CHECK_CUDNN(cudnnSetStream(cur_handle, cur_stream));

    // Build the descriptor for x
    int64_t xDim[] = {n, c, h, w};
    int64_t xStr[] = {c * h * w, h * w, w, 1};
    int64_t xUi = 'x';
    int64_t alignment = 8;
    cudnnBackendDescriptor_t xDesc;
    CHECK_CUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_TENSOR_DESCRIPTOR, &xDesc));
    cudnnDataType_t xDtype = cudnnDataType_t(tx);
    CHECK_CUDNN(cudnnBackendSetAttribute(xDesc, CUDNN_ATTR_TENSOR_DATA_TYPE,
                            CUDNN_TYPE_DATA_TYPE, 1, &xDtype));
    CHECK_CUDNN(cudnnBackendSetAttribute(xDesc, CUDNN_ATTR_TENSOR_DIMENSIONS,
                            CUDNN_TYPE_INT64, 4, xDim));
    CHECK_CUDNN(cudnnBackendSetAttribute(xDesc, CUDNN_ATTR_TENSOR_STRIDES,
                            CUDNN_TYPE_INT64, 4, xStr));
    CHECK_CUDNN(cudnnBackendSetAttribute(xDesc, CUDNN_ATTR_TENSOR_UNIQUE_ID,
                            CUDNN_TYPE_INT64, 1, &xUi));
    CHECK_CUDNN(cudnnBackendSetAttribute(xDesc, CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT,
                            CUDNN_TYPE_INT64, 1, &alignment));
    CHECK_CUDNN(cudnnBackendFinalize(xDesc));

    // Build the descriptor for w
    int64_t wDim[] = {k, c, r, s};
    int64_t wStr[] = {c * r * s, r * s, s, 1};
    int64_t wUi = 'w';
    cudnnBackendDescriptor_t wDesc;
    CHECK_CUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_TENSOR_DESCRIPTOR, &wDesc));
    cudnnDataType_t wDtype = cudnnDataType_t(tw);
    CHECK_CUDNN(cudnnBackendSetAttribute(wDesc, CUDNN_ATTR_TENSOR_DATA_TYPE,
                            CUDNN_TYPE_DATA_TYPE, 1, &wDtype));
    CHECK_CUDNN(cudnnBackendSetAttribute(wDesc, CUDNN_ATTR_TENSOR_DIMENSIONS,
                            CUDNN_TYPE_INT64, 4, wDim));
    CHECK_CUDNN(cudnnBackendSetAttribute(wDesc, CUDNN_ATTR_TENSOR_STRIDES,
                            CUDNN_TYPE_INT64, 4, wStr));
    CHECK_CUDNN(cudnnBackendSetAttribute(wDesc, CUDNN_ATTR_TENSOR_UNIQUE_ID,
                            CUDNN_TYPE_INT64, 1, &wUi));
    CHECK_CUDNN(cudnnBackendSetAttribute(wDesc, CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT,
                            CUDNN_TYPE_INT64, 1, &alignment));
    CHECK_CUDNN(cudnnBackendFinalize(wDesc));

    // Build the descriptor for y
    int64_t yDim[] = {n, k, p, q};
    int64_t yStr[] = {k * p * q, p * q, q, 1};
    int64_t yUi = 'y';
    cudnnBackendDescriptor_t yDesc;
    CHECK_CUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_TENSOR_DESCRIPTOR, &yDesc));
    cudnnDataType_t yDtype = cudnnDataType_t(ty);
    CHECK_CUDNN(cudnnBackendSetAttribute(yDesc, CUDNN_ATTR_TENSOR_DATA_TYPE,
                            CUDNN_TYPE_DATA_TYPE, 1, &yDtype));
    CHECK_CUDNN(cudnnBackendSetAttribute(yDesc, CUDNN_ATTR_TENSOR_DIMENSIONS,
                            CUDNN_TYPE_INT64, 4, yDim));
    CHECK_CUDNN(cudnnBackendSetAttribute(yDesc, CUDNN_ATTR_TENSOR_STRIDES,
                            CUDNN_TYPE_INT64, 4, yStr));
    CHECK_CUDNN(cudnnBackendSetAttribute(yDesc, CUDNN_ATTR_TENSOR_UNIQUE_ID,
                            CUDNN_TYPE_INT64, 1, &yUi));
    CHECK_CUDNN(cudnnBackendSetAttribute(yDesc, CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT,
                            CUDNN_TYPE_INT64, 1, &alignment));
    CHECK_CUDNN(cudnnBackendFinalize(yDesc));

    // Build the descriptor for the convolution operator
    cudnnBackendDescriptor_t cDesc;
    int64_t nbDims = 2;
    cudnnDataType_t compType = cudnnDataType_t(compute_type);
    cudnnConvolutionMode_t mode = CUDNN_CROSS_CORRELATION;
    int64_t pad[] = {pad_dim1, pad_dim2};
    int64_t filterStr[] = {str_dim1, str_dim2};
    int64_t dilation[] = {dil_dim1, dil_dim2};
    CHECK_CUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR, &cDesc));
    CHECK_CUDNN(cudnnBackendSetAttribute(cDesc, CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS,
                            CUDNN_TYPE_INT64, 1, &nbDims));
    CHECK_CUDNN(cudnnBackendSetAttribute(cDesc, CUDNN_ATTR_CONVOLUTION_COMP_TYPE,
                            CUDNN_TYPE_DATA_TYPE, 1, &compType));
    CHECK_CUDNN(cudnnBackendSetAttribute(cDesc, CUDNN_ATTR_CONVOLUTION_CONV_MODE,
                            CUDNN_TYPE_CONVOLUTION_MODE, 1, &mode));
    CHECK_CUDNN(cudnnBackendSetAttribute(cDesc, CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS,
                            CUDNN_TYPE_INT64, nbDims, pad));
    CHECK_CUDNN(cudnnBackendSetAttribute(cDesc, CUDNN_ATTR_CONVOLUTION_POST_PADDINGS,
                            CUDNN_TYPE_INT64, nbDims, pad));
    CHECK_CUDNN(cudnnBackendSetAttribute(cDesc, CUDNN_ATTR_CONVOLUTION_DILATIONS,
                            CUDNN_TYPE_INT64, nbDims, dilation));
    CHECK_CUDNN(cudnnBackendSetAttribute(cDesc, CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES,
                            CUDNN_TYPE_INT64, nbDims, filterStr));
    CHECK_CUDNN(cudnnBackendFinalize(cDesc));

    // Build the descriptor for the convolution forward operation
    cudnnBackendDescriptor_t fprop;
    void *p_alpha = nullptr;
    void *p_beta = nullptr;
    set_alpha_beta(&p_alpha, &p_beta, compType);
    CHECK_CUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR,
                &fprop));
    CHECK_CUDNN(cudnnBackendSetAttribute(fprop, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X,
                CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &xDesc));
    CHECK_CUDNN(cudnnBackendSetAttribute(fprop, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W,
                CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &wDesc));
    CHECK_CUDNN(cudnnBackendSetAttribute(fprop, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y,
                CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &yDesc));
    CHECK_CUDNN(cudnnBackendSetAttribute(fprop,
                CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC,
                CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &cDesc));
    CHECK_CUDNN(cudnnBackendSetAttribute(fprop, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA,
                get_attribute_type_from_compute_type(compType), 1, p_alpha));
    CHECK_CUDNN(cudnnBackendSetAttribute(fprop, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA,
                get_attribute_type_from_compute_type(compType), 1, p_beta));
    CHECK_CUDNN(cudnnBackendFinalize(fprop));

    // Build the operation graph descriptor
    cudnnBackendDescriptor_t op_graph;
    CHECK_CUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR, &op_graph));
    CHECK_CUDNN(cudnnBackendSetAttribute(op_graph, CUDNN_ATTR_OPERATIONGRAPH_OPS,
                CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &fprop));
    CHECK_CUDNN(cudnnBackendSetAttribute(op_graph, CUDNN_ATTR_OPERATIONGRAPH_HANDLE,
                CUDNN_TYPE_HANDLE, 1, &cur_handle));
    CHECK_CUDNN(cudnnBackendFinalize(op_graph));

    // Set up engine config
    cudnnBackendDescriptor_t engine;
    CHECK_CUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_ENGINE_DESCRIPTOR, &engine));
    CHECK_CUDNN(cudnnBackendSetAttribute(engine, CUDNN_ATTR_ENGINE_OPERATION_GRAPH,
                CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &op_graph));
    // TODO: Is it okay to hardcode the engine to be CUDNN_ATTR_ENGINE_GLOBAL_INDEX 0?
    // As mentioned here: https://docs.nvidia.com/deeplearning/cudnn/developer/graph-api.html,
    // Engine selection should be determined based on some heuristics.
    int64_t gidx = 0;
    CHECK_CUDNN(cudnnBackendSetAttribute(engine, CUDNN_ATTR_ENGINE_GLOBAL_INDEX,
                CUDNN_TYPE_INT64, 1, &gidx));
    CHECK_CUDNN(cudnnBackendFinalize(engine));

    cudnnBackendDescriptor_t engcfg;
    CHECK_CUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_ENGINECFG_DESCRIPTOR, &engcfg));
    CHECK_CUDNN(cudnnBackendSetAttribute(engcfg, CUDNN_ATTR_ENGINECFG_ENGINE,
                CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &engine));
    CHECK_CUDNN(cudnnBackendFinalize(engcfg));

    // Set up the execution plan
    cudnnBackendDescriptor_t plan;
    CHECK_CUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR, &plan));
    CHECK_CUDNN(cudnnBackendSetAttribute(plan, CUDNN_ATTR_EXECUTION_PLAN_HANDLE, CUDNN_TYPE_HANDLE, 1, &cur_handle));
    CHECK_CUDNN(cudnnBackendSetAttribute(plan, CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,
                CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &engcfg));
    CHECK_CUDNN(cudnnBackendFinalize(plan));

    int64_t workspaceSize;
    CHECK_CUDNN(cudnnBackendGetAttribute(plan, CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE,
                CUDNN_TYPE_INT64, 1, NULL, &workspaceSize));
    
    void *dev_ptrs[3] = {ptr_x, ptr_w, ptr_y}; // device pointers
    int64_t uids[3] = {'x', 'w', 'y'};
    void *workspace = request_cuda_workspace(workspaceSize, false);

    cudnnBackendDescriptor_t varpack;
    CHECK_CUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR, &varpack));
    CHECK_CUDNN(cudnnBackendSetAttribute(varpack, CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS,
                CUDNN_TYPE_VOID_PTR, 3, dev_ptrs));
    CHECK_CUDNN(cudnnBackendSetAttribute(varpack, CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS,
                CUDNN_TYPE_INT64, 3, uids));
    CHECK_CUDNN(cudnnBackendSetAttribute(varpack, CUDNN_ATTR_VARIANT_PACK_WORKSPACE,
                CUDNN_TYPE_VOID_PTR, 1, &workspace));
    CHECK_CUDNN(cudnnBackendFinalize(varpack));

    // Execute the plan
    CHECK_CUDNN(cudnnBackendExecute(cur_handle, plan, varpack));

    // Cleanup
    hidet_cuda_free_async(workspace, cur_stream);
    CHECK_CUDNN(cudnnBackendDestroyDescriptor(xDesc));
    CHECK_CUDNN(cudnnBackendDestroyDescriptor(wDesc));
    CHECK_CUDNN(cudnnBackendDestroyDescriptor(yDesc));
    CHECK_CUDNN(cudnnBackendDestroyDescriptor(cDesc));
    CHECK_CUDNN(cudnnBackendDestroyDescriptor(fprop));
    CHECK_CUDNN(cudnnBackendDestroyDescriptor(op_graph));
    CHECK_CUDNN(cudnnBackendDestroyDescriptor(engine));
    CHECK_CUDNN(cudnnBackendDestroyDescriptor(engcfg));
    CHECK_CUDNN(cudnnBackendDestroyDescriptor(plan));
}

