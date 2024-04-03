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
from typing import Union, List
from hidet.ir.dtypes import DataType
from hidet.ir.type import void_p
from hidet.ffi.array import Array
from .utils import as_pointer, as_type_code
from .ffi import cublasComputeType, cudaDataType
from . import ffi


def gemm(
    m: int,
    n: int,
    k: int,
    type_a: Union[int, cudaDataType, DataType],
    type_b: Union[int, cudaDataType, DataType],
    type_c: Union[int, cudaDataType, DataType],
    a,
    b,
    c,
    trans_a: bool,
    trans_b: bool,
    compute_type: Union[int, cublasComputeType],
):
    """
    Matrix multiplication of two matrices using cublas in row major by default.

    The matrix of A, B, and C are stored in row-major order (if not transposed).

        A: m x k
        B: k x n
        C: m x n


    Parameters
    ----------
    m: int
        Number of rows of matrix op(A) and of matrix C.
    n: int
        Number of columns of matrix op(B) and of matrix C.
    k: int
        Number of columns of matrix op(A) and of rows of matrix op(B).
    type_a: Union[int, cudaDataType, DataType]
        Type of elements in matrix A.
    type_b: Union[int, cudaDataType, DataType]
        Type of elements in matrix B.
    type_c: Union[int, cudaDataType, DataType]
        Type of elements in matrix C.
    a: hidet.Tensor or int
        Matrix A, can be either a Tensor or an integer (the address of the matrix).
    b: hidet.Tensor or int
        Matrix B, can be either a Tensor or an integer (the address of the matrix).
    c: hidet.Tensor or int
        Matrix C, can be either a Tensor or an integer (the address of the matrix).
    trans_a: bool
        Whether matrix A is transposed.
    trans_b: bool
        Whether matrix B is transposed.
    compute_type: Union[int, cublasComputeType]
        The compute type of the operation.
    """
    ffi.gemm(
        m,
        n,
        k,
        as_type_code(type_a),
        as_type_code(type_b),
        as_type_code(type_c),
        as_pointer(a),
        as_pointer(b),
        as_pointer(c),
        trans_a,
        trans_b,
        compute_type,
    )


def strided_gemm(
    bs: int,
    m: int,
    n: int,
    k: int,
    type_a: Union[int, cudaDataType, DataType],
    type_b: Union[int, cudaDataType, DataType],
    type_c: Union[int, cudaDataType, DataType],
    a,
    b,
    c,
    stride_a: int,
    stride_b: int,
    stride_c: int,
    trans_a: bool,
    trans_b: bool,
    compute_type: Union[int, cublasComputeType],
):
    """
    Batch matrix multiplication of two matrices using cublas in row major order by default.

    The matrix of A, B, and C are stored in row-major order (if not transposed).

        A: bs x m x k
        B: bs x k x n
        C: bs x m x n


    Parameters
    ----------
    bs: int
        Batch size.
    m: int
        Number of rows of matrix op(A) and of matrix C.
    n: int
        Number of columns of matrix op(B) and of matrix C.
    k: int
        Number of columns of matrix op(A) and of rows of matrix op(B).
    type_a: Union[int, DataType]
        Type of elements in matrix A.
    type_b: Union[int, DataType]
        Type of elements in matrix B.
    type_c: Union[int, DataType]
        Type of elements in matrix C.
    a: Tensor or int
        Matrix A, can be either a Tensor or an integer (the address of the matrix).
    b: Tensor or int
        Matrix B, can be either a Tensor or an integer (the address of the matrix).
    c: Tensor or int
        Matrix C, can be either a Tensor or an integer (the address of the matrix).
    stride_a: int
        Stride of matrix A on batch dimension.
    stride_b: int
        Stride of matrix B on batch dimension.
    stride_c: int
        Stride of matrix C on batch dimension.
    trans_a: bool
        Whether matrix A is transposed.
    trans_b: bool
        Whether matrix B is transposed.
    compute_type: Union[int, cublasComputeType]
        The compute type of the operation.
    """
    ffi.strided_gemm(
        bs,
        m,
        n,
        k,
        as_type_code(type_a),
        as_type_code(type_b),
        as_type_code(type_c),
        as_pointer(a),
        as_pointer(b),
        as_pointer(c),
        stride_a,
        stride_b,
        stride_c,
        trans_a,
        trans_b,
        compute_type,
    )


def batched_gemm(
    bs: int,
    m: int,
    n: int,
    k: int,
    type_a: Union[int, cudaDataType, DataType],
    type_b: Union[int, cudaDataType, DataType],
    type_c: Union[int, cudaDataType, DataType],
    a: Union[Array, List],
    b: Union[Array, List],
    c: Union[Array, List],
    trans_a: bool,
    trans_b: bool,
    compute_type: Union[int, cublasComputeType],
):
    """
    Batch matrix multiplication of two matrices using cublas in row major order by default.

    The matrix of A, B, and C are stored as arrays where each array element is one matrix in
    row-major order (if not transposed), and the length of the array is the batch size.

        A: bs x m x k
        B: bs x k x n
        C: bs x m x n

    Parameters
    ----------
    bs: int
        Batch size.
    m: int
        Number of rows of matrix op(A) and of matrix C.
    n: int
        Number of columns of matrix op(B) and of matrix C.
    k: int
        Number of columns of matrix op(A) and of rows of matrix op(B).
    type_a: Union[int, DataType]
        Type of elements in matrix A.
    type_b: Union[int, DataType]
        Type of elements in matrix B.
    type_c: Union[int, DataType]
        Type of elements in matrix C.
    a: hidet.ffi.array.Array or List[Tensor]
        Matrix A, can be either a list of Tensors or an Array object constructed from a list of Tensors.
    b: hidet.ffi.array.Array or List[Tensor]
        Matrix B, can be either a list of Tensors or an Array object constructed from a list of Tensors.
    c: hidet.ffi.array.Array or List[Tensor]
        Matrix C, can be either a list of Tensors or an Array object constructed from a list of Tensors.
    trans_a: bool
        Whether matrix A is transposed.
    trans_b: bool
        Whether matrix B is transposed.
    compute_type: Union[int, cublasComputeType]
        The compute type of the operation.
    """

    def convert_list_to_array(l):
        ret = Array(void_p, len(l))
        for i in range(len(l)):
            ret[i] = l[i].storage.addr
        return ret

    if isinstance(a, List):
        a = convert_list_to_array(a)
    if isinstance(b, List):
        b = convert_list_to_array(b)
    if isinstance(c, List):
        c = convert_list_to_array(c)

    ffi.batched_gemm(
        bs,
        m,
        n,
        k,
        as_type_code(type_a),
        as_type_code(type_b),
        as_type_code(type_c),
        a.data_ptr(),
        b.data_ptr(),
        c.data_ptr(),
        trans_a,
        trans_b,
        compute_type,
    )
