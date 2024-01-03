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
from hidet.ir.dtypes import int32, boolean
from hidet.ir.type import FuncType, void_p, void
from hidet.ir.primitives.func import register_primitive_function
from hidet.utils import initialize


@initialize()
def register_cublas_kernels():
    register_primitive_function(
        name='cublas.gemm',
        func_or_type=FuncType(
            param_types=[
                int32,  # m
                int32,  # n
                int32,  # k
                int32,  # type_a (cudaDataType)
                int32,  # type_b (cudaDataType)
                int32,  # type_c (cudaDataType)
                void_p,  # a
                void_p,  # b
                void_p,  # c
                boolean,  # trans_a
                boolean,  # trans_b
                int32,  # compute_type (cublasComputeType)
            ],
            ret_type=void,
        ),
        codegen_name='hidet_cublas_gemm',
    )
    register_primitive_function(
        name='cublas.bgemm',
        func_or_type=FuncType(
            param_types=[
                int32,  # bs
                int32,  # m
                int32,  # n
                int32,  # k
                int32,  # type_a (cudaDataType)
                int32,  # type_b (cudaDataType)
                int32,  # type_c (cudaDataType)
                void_p,  # a
                void_p,  # b
                void_p,  # c
                boolean,  # trans_a
                boolean,  # trans_b
                int32,  # compute_type (cublasComputeType)
            ],
            ret_type=void,
        ),
        codegen_name='hidet_cublas_bgemm',
    )
