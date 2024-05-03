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
):
    """
    Initialize a CUDA tensor map construct for async tensor bulk copy.

    See Also:
    https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#using-tma-to-transfer-multi-dimensional-arrays

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
    """

    dtypes = ['float16', 'float32', 'float64', 'int32', 'int64', 'uint32', 'uint64']
    assert str(dtype) in dtypes

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
    CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
    CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
    CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
    CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
);
    """

    return BlackBoxStmt(template_string, tensor_map, rank, tensor_ptr, size, stride, box_size, elem_stride)
