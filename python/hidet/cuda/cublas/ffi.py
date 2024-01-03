# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
import glob
from enum import IntEnum
from ctypes import c_int32, c_int64, c_void_p, c_bool, c_char_p
from hidet.ffi.ffi import get_func
from hidet.utils.py import initialize


class cudaDataType(IntEnum):
    """
    See Also: https://docs.nvidia.com/cuda/cublas/index.html#cudadatatype-t
    """

    CUDA_R_16F = 2
    CUDA_C_16F = 6
    CUDA_R_16BF = 14
    CUDA_C_16BF = 15
    CUDA_R_32F = 0
    CUDA_C_32F = 4
    CUDA_R_64F = 1
    CUDA_C_64F = 5
    CUDA_R_4I = 16
    CUDA_C_4I = 17
    CUDA_R_4U = 18
    CUDA_C_4U = 19
    CUDA_R_8I = 3
    CUDA_C_8I = 7
    CUDA_R_8U = 8
    CUDA_C_8U = 9
    CUDA_R_16I = 20
    CUDA_C_16I = 21
    CUDA_R_16U = 22
    CUDA_C_16U = 23
    CUDA_R_32I = 10
    CUDA_C_32I = 11
    CUDA_R_32U = 12
    CUDA_C_32U = 13
    CUDA_R_64I = 24
    CUDA_C_64I = 25
    CUDA_R_64U = 26
    CUDA_C_64U = 27
    CUDA_R_8F_E4M3 = 28  # real as a nv_fp8_e4m3
    CUDA_R_8F_E5M2 = 29  # real as a nv_fp8_e5m2


class cublasComputeType(IntEnum):
    """
    See Also: https://docs.nvidia.com/cuda/cublas/index.html#cublascomputetype-t
    """

    CUBLAS_COMPUTE_16F = 64  # half - default
    CUBLAS_COMPUTE_16F_PEDANTIC = 65  # half - pedantic
    CUBLAS_COMPUTE_32F = 68  # float - default
    CUBLAS_COMPUTE_32F_PEDANTIC = 69  # float - pedantic
    CUBLAS_COMPUTE_32F_FAST_16F = 74  # float - fast allows down-converting inputs to half or TF32
    CUBLAS_COMPUTE_32F_FAST_16BF = 75  # float - fast allows down-converting inputs to bfloat16 or TF32
    CUBLAS_COMPUTE_32F_FAST_TF32 = 77  # float - fast allows down-converting inputs to TF32
    CUBLAS_COMPUTE_64F = 70  # double - default
    CUBLAS_COMPUTE_64F_PEDANTIC = 71  # double - pedantic
    CUBLAS_COMPUTE_32I = 72  # signed 32-bit int - default
    CUBLAS_COMPUTE_32I_PEDANTIC = 73  # signed 32-bit int - pedantic


set_library_path = get_func(func_name='hidet_cublas_set_library_path', arg_types=[c_char_p], restype=None)  # path

gemm = get_func(
    func_name='hidet_cublas_gemm',
    arg_types=[
        c_int32,  # m
        c_int32,  # n
        c_int32,  # k
        c_int32,  # type a
        c_int32,  # type b
        c_int32,  # type c
        c_void_p,  # ptr a
        c_void_p,  # ptr b
        c_void_p,  # ptr c
        c_bool,  # trans a
        c_bool,  # trans b
        c_int32,  # compute type
    ],
    restype=None,
)

strided_gemm = get_func(
    func_name='hidet_cublas_strided_gemm',
    arg_types=[
        c_int32,  # batch size
        c_int32,  # m
        c_int32,  # n
        c_int32,  # k
        c_int32,  # type a
        c_int32,  # type b
        c_int32,  # type c
        c_void_p,  # ptr a
        c_void_p,  # ptr b
        c_void_p,  # ptr c
        c_int64,  # stride a
        c_int64,  # stride b
        c_int64,  # stride c
        c_bool,  # trans a
        c_bool,  # trans b
        c_int32,  # compute type
    ],
    restype=None,
)


@initialize()
def set_cublas_library_path():
    # use nvidia-cuda-cublas
    for path in sys.path:
        nvidia_path = os.path.join(path, 'nvidia')
        if not os.path.exists(nvidia_path):
            continue
        cublas_path = glob.glob(os.path.join(nvidia_path, 'cublas', 'lib', 'libcublas.so.[0-9]*'))
        if cublas_path:
            set_library_path(cublas_path[0].encode('utf-8'))
            return
