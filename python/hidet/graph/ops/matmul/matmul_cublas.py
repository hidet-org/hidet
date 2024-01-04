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
from typing import Union, List, Optional

import hidet
from hidet.ir.module import IRModule
from hidet.ir.type import DataType
from hidet.ir.expr import Expr, is_true
from hidet.ir.dtypes import f16, f32
from hidet.utils import prod
from hidet.cuda.cublas import cublasComputeType
from ..utils import Task, Operator, Tensor, input_like
from ..utils import TensorInput


class CublasMatmulTask(Task):
    def __init__(self, a: TensorInput, b: TensorInput, compute_type: Optional[Union[int, cublasComputeType]] = None):
        from hidet.ir.compute import cops

        # check
        if a.type.dtype != b.type.dtype:
            raise ValueError('dtype of a and b must be the same, got {} and {}'.format(a.type.dtype, b.type.dtype))

        self.compute_type: cublasComputeType = self.resolve_compute_type(a.type.dtype, a.type.dtype, compute_type)

        c = cops.matmul(a, b, allow_1d=True)
        super().__init__(
            name='cublas_matmul', inputs=[a, b], outputs=[c], attributes={'compute_type': self.compute_type}
        )

    def resolve_compute_type(
        self, in_dtype: DataType, out_dtype: DataType, compute_type: Optional[Union[int, cublasComputeType]]
    ) -> cublasComputeType:
        if compute_type is not None:
            return cublasComputeType(compute_type)
        if in_dtype == out_dtype == f16:
            # use tensor core whenever possible
            return cublasComputeType.CUBLAS_COMPUTE_16F
        elif in_dtype == out_dtype == f32:
            # use tensor core whenever possible
            return cublasComputeType.CUBLAS_COMPUTE_32F
        else:
            raise NotImplementedError(
                'not implemented resolve rules for compute_type with in_dtype={}, out_dtype={}'.format(
                    in_dtype, out_dtype
                )
            )

    def convert_to_strided_gemm(self, a_shape: List[Expr], b_shape: List[Expr], c_shape: List[Expr]):
        a_rank: int = len(a_shape)
        b_rank: int = len(b_shape)

        assert a_rank >= 1 and b_rank >= 1 and (a_rank >= 2 or b_rank >= 2)
        if a_rank == 1:
            bs = prod(b_shape[:-2])
            m = 1
            n = b_shape[-1]
            k = a_shape[0]
            stride_a = 0
            stride_b = b_shape[-2] * b_shape[-1]
            stride_c = c_shape[-2] * c_shape[-1]
        elif b_rank == 1:
            bs = prod(a_shape[:-2])
            m = a_shape[-2]
            n = 1
            k = b_shape[0]
            stride_a = a_shape[-2] * a_shape[-1]
            stride_b = 0
            stride_c = c_shape[-1]
        else:
            if is_true(prod(a_shape[:-2]) == 1):
                bs = prod(b_shape[:-2])
                m = a_shape[-2]
                n = b_shape[-1]
                k = a_shape[-1]
                stride_a = 0
                stride_b = b_shape[-2] * b_shape[-1]
                stride_c = c_shape[-2] * c_shape[-1]
            elif is_true(prod(b_shape[:-2]) == 1):
                bs = prod(a_shape[:-2])
                m = a_shape[-2]
                n = b_shape[-1]
                k = a_shape[-1]
                stride_a = a_shape[-2] * a_shape[-1]
                stride_b = 0
                stride_c = c_shape[-2] * c_shape[-1]
            elif all(is_true(a == b) for a, b in zip(a_shape[:-2], b_shape[:-2])):
                bs = prod(a_shape[:-2])
                m = a_shape[-2]
                n = b_shape[-1]
                k = a_shape[-1]
                stride_a = a_shape[-2] * a_shape[-1]
                stride_b = b_shape[-2] * b_shape[-1]
                stride_c = c_shape[-2] * c_shape[-1]
            else:
                # todo: add cublasGemmBatchedEx to support this case
                # https://docs.nvidia.com/cuda/cublas/index.html#cublasgemmbatchedex
                raise NotImplementedError('Can not convert matmul {} @ {} to strided_gemm'.format(a_shape, b_shape))
        return bs, m, n, k, stride_a, stride_b, stride_c

    def implement_cuda(self, working_dir: str) -> IRModule:
        from hidet.lang import attrs
        from hidet.lang.cuda import cublas

        dtype = self.inputs[0].type.dtype
        c_dtype = self.outputs[0].type.dtype
        a_shape = list(self.inputs[0].type.shape)
        b_shape = list(self.inputs[1].type.shape)
        c_shape = list(self.outputs[0].type.shape)

        with hidet.script_module() as script_module:

            def generate(a: Expr, b: Expr, c: Expr) -> Expr:
                bs, m, n, k, stride_a, stride_b, stride_c = self.convert_to_strided_gemm(a_shape, b_shape, c_shape)
                return cublas.strided_gemm(
                    bs,
                    m,
                    n,
                    k,
                    dtype,
                    dtype,
                    c_dtype,
                    a,
                    b,
                    c,
                    stride_a,
                    stride_b,
                    stride_c,
                    False,
                    False,
                    self.compute_type,
                )

            @hidet.script
            def launch(a: dtype[a_shape], b: dtype[b_shape], c: c_dtype[c_shape]):
                attrs.func_kind = 'public'

                generate(a, b, c)

        return script_module.ir_module()


class CublasMatmulOp(Operator):
    def __init__(self, a: Tensor, b: Tensor):
        task = CublasMatmulTask(input_like(a, 'a'), input_like(b, 'b'))
        super().__init__(inputs=[a, b], attributes={}, task=task)


def matmul_cublas(a: Tensor, b: Tensor) -> Tensor:
    return CublasMatmulOp(a, b).outputs[0]
