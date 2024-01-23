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
from typing import Union, Optional

import hidet
from hidet.ir.module import IRModule
from hidet.ir.expr import Expr
from hidet.cuda.cublas import cublasComputeType
from ..utils import Task, Operator, Tensor, input_like
from ..utils import TensorInput
from ..utils.schedule_utils import convert_to_cublas_strided_gemm, resolve_cublas_compute_type


class CublasMatmulTask(Task):
    def __init__(self, a: TensorInput, b: TensorInput, compute_type: Optional[Union[int, cublasComputeType]] = None):
        from hidet.ir.compute import cops

        # check
        if a.type.dtype != b.type.dtype:
            raise ValueError('dtype of a and b must be the same, got {} and {}'.format(a.type.dtype, b.type.dtype))

        self.compute_type: cublasComputeType = resolve_cublas_compute_type(a.type.dtype, a.type.dtype, compute_type)

        c = cops.matmul(a, b, allow_1d=True)
        super().__init__(
            name='cublas_matmul', inputs=[a, b], outputs=[c], attributes={'compute_type': self.compute_type}
        )

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
                bs, m, n, k, stride_a, stride_b, stride_c = convert_to_cublas_strided_gemm(a_shape, b_shape, c_shape)
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
