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
from typing import Union
from hidet.ir.type import DataType
from hidet.ir.expr import Expr
from hidet.ir.stmt import BlackBoxStmt


def create_tensor_map(
    tensor_map: Expr,
    dtype: Union[DataType, str],
    rank: Expr,
    tensor_ptr: Expr,
    size: Expr,
    stride: Expr,
    box_size: Expr,
    elem_stride: Expr,
    interleave: str = 'NONE',
    swizzle: str = 'NONE',
    l2_promotion: str = 'NONE',
):
    """
    Initialize a CUDA tensor map construct for async tensor bulk copy.

    See Also:
    https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#using-tma-to-transfer-multi-dimensional-arrays
    https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html

    Parameters
    ----------
    name: str
        The name of the tensor map, must be unique
    dtype: str
        Data type of the tensor to be copied, supported types listed above
    rank: int
        The number of dimensions of the tensor
    tensor_ptr: Expr
        A pointer to the first element of the tensor
    size: int[rank]
        The number of elements along each axis
    stride: int[rank - 1]
        The stride is the number of bytes to traverse from the first element of one row to the next.
        It must be a multiple of 16.
    box_size: int[rank]:
        The box_size is the size of the shared memory buffer that is used as the
        destination of a TMA transfer.
    elem_stride: int[rank]
        The distance between elements in units of sizeof(element). A stride of 2
        can be used to load only the real component of a complex-valued tensor, for instance
    swizzle_mode: str
        The shared memory bank swizzling pattern. It has to be one of the following:
        - NONE: No swizzling
        - 32B: 32-byte swizzling
        - 64B: 64-byte swizzling
        - 128B: 128-byte swizzling
    """

    if str(dtype) in ('float8_e5m2', 'float8_e4m3', 'f8e5m2', 'f8e4m3'):
        dtype = 'uint8'

    dtypes = [
        'float16',
        'bfloat16',
        'float32',
        'float64',
        'uint8',
        'uint16',
        'uint32',
        'int32',
        'int64',
        'uint32',
        'uint64',
    ]
    assert str(dtype) in dtypes

    # From the CUDA driver API(linked above):
    # ```
    # typedef enum CUtensorMapSwizzle_enum {
    #           CU_TENSOR_MAP_SWIZZLE_NONE = 0,
    #           CU_TENSOR_MAP_SWIZZLE_32B,
    #           CU_TENSOR_MAP_SWIZZLE_64B,
    #           CU_TENSOR_MAP_SWIZZLE_128B
    #       } CUtensorMapSwizzle;
    # ```
    swizzle_modes = ('NONE', '32B', '64B', '128B')
    assert swizzle in swizzle_modes, f"Invalid swizzle mode: must be one of {swizzle_modes}, but got {swizzle}"

    # From the CUDA driver API(linked above):
    # ```
    # typedef enum CUtensorMapInterleave_enum {
    #           CU_TENSOR_MAP_INTERLEAVE_NONE = 0,
    #           CU_TENSOR_MAP_INTERLEAVE_16B,
    #           CU_TENSOR_MAP_INTERLEAVE_32B
    #       } CUtensorMapInterleave;
    # ```
    interleave_modes = ('NONE', '16B', '32B')
    assert (
        interleave in interleave_modes
    ), f"Invalid interleave mode: must be one of {interleave_modes}, but got {interleave}"

    # From the CUDA driver API(linked above):
    # ```
    # typedef enum CUtensorMapL2promotion_enum {
    #           CU_TENSOR_MAP_L2_PROMOTION_NONE = 0,
    #           CU_TENSOR_MAP_L2_PROMOTION_L2_64B,
    #           CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
    #           CU_TENSOR_MAP_L2_PROMOTION_L2_256B
    #       } CUtensorMapL2promotion;
    # ```
    l2_promotion_modes = ('NONE', '64B', '128B', '256B')
    assert (
        l2_promotion in l2_promotion_modes
    ), f"Invalid L2 promotion mode: must be one of {l2_promotion_modes}, but got {l2_promotion}"
    if l2_promotion != 'NONE':
        l2_promotion = 'L2_' + l2_promotion

    template_string = f"""
cuTensorMapEncodeTiled(
    {{}},
    CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_{dtype.upper()},
    {{}}, 
    {{}}, 
    {{}}, 
    {{}}, 
    {{}}, 
    {{}}, 
    CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_{interleave},
    CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_{swizzle},
    CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_{l2_promotion},
    CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
);
    """

    return BlackBoxStmt(template_string, tensor_map, rank, tensor_ptr, size, stride, box_size, elem_stride)
