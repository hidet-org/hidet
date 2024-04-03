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
from ctypes import c_int32, c_void_p, c_char_p
from hidet.ffi.ffi import get_func

from hidet.utils.py import initialize


class cudnnDataType(IntEnum):
    """
    defined in cudnn_ops_infer_v8.h
    """

    CUDNN_DATA_FLOAT = 0
    CUDNN_DATA_DOUBLE = 1
    CUDNN_DATA_HALF = 2
    CUDNN_DATA_INT8 = 3
    CUDNN_DATA_INT32 = 4
    CUDNN_DATA_INT8x4 = 5
    CUDNN_DATA_UINT8 = 6
    CUDNN_DATA_UINT8x4 = 7
    CUDNN_DATA_INT8x32 = 8
    CUDNN_DATA_BFLOAT16 = 9
    CUDNN_DATA_INT64 = 10


set_library_path = get_func(func_name='hidet_cudnn_set_library_path', arg_types=[c_char_p], restype=None)

conv2d = get_func(
    func_name='hidet_cudnn_conv2d',
    arg_types=[
        c_int32,  # n
        c_int32,  # c
        c_int32,  # h
        c_int32,  # w
        c_int32,  # k
        c_int32,  # r
        c_int32,  # s
        c_int32,  # p
        c_int32,  # q
        c_void_p,  # ptr_x
        c_void_p,  # ptr_w
        c_void_p,  # ptr_y
        c_int32,  # tx
        c_int32,  # tw
        c_int32,  # ty
        c_int32,  # compute_type
        c_int32,  # pad_dim1
        c_int32,  # pad_dim2
        c_int32,  # str_dim1
        c_int32,  # str_dim2
        c_int32,  # dil_dim1
        c_int32,  # dil_dim2
    ],
    restype=None,
)

conv2d_gemm = get_func(
    func_name='hidet_cudnn_conv2d_gemm',
    arg_types=[
        c_int32,  # n
        c_int32,  # c
        c_int32,  # h
        c_int32,  # w
        c_int32,  # k
        c_int32,  # r
        c_int32,  # s
        c_void_p,  # ptr_x
        c_void_p,  # ptr_w
        c_void_p,  # ptr_y
        c_int32,  # tx
        c_int32,  # tw
        c_int32,  # ty
        c_int32,  # compute_type
        c_int32,  # pad_dim1
        c_int32,  # pad_dim2
        c_int32,  # str_dim1
        c_int32,  # str_dim2
        c_int32,  # dil_dim1
        c_int32,  # dil_dim2
    ],
    restype=None,
)

conv2d_autoselect_algo = get_func(
    func_name='hidet_cudnn_conv2d_autoselect_algo',
    arg_types=[
        c_int32,  # n
        c_int32,  # c
        c_int32,  # h
        c_int32,  # w
        c_int32,  # k
        c_int32,  # r
        c_int32,  # s
        c_void_p,  # ptr_x
        c_void_p,  # ptr_w
        c_void_p,  # ptr_y
        c_int32,  # tx
        c_int32,  # tw
        c_int32,  # ty
        c_int32,  # compute_type
        c_int32,  # pad_dim1
        c_int32,  # pad_dim2
        c_int32,  # str_dim1
        c_int32,  # str_dim2
        c_int32,  # dil_dim1
        c_int32,  # dil_dim2
    ],
    restype=None,
)


@initialize()
def set_cudnn_library_path():
    # use nvidia-cuda-cudnn
    for path in sys.path:
        nvidia_path = os.path.join(path, 'nvidia')
        if not os.path.exists(nvidia_path):
            continue
        cudnn_path = glob.glob(os.path.join(nvidia_path, 'cudnn', 'lib', 'libcudnn.so.[0-9]*'))
        if cudnn_path:
            set_library_path(cudnn_path[0].encode('utf-8'))
            return
