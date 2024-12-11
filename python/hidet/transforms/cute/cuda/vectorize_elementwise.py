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
"""
The VectorizeElementwise class and its associated pass aim to optimize element-wise arithmetic operations by
vectorizing them. This optimization is particularly beneficial for improving performance on hardware that supports
vectorized operations.

Classes:
    - VectorizeElementwise: Rewrites arithmetic operations to apply vectorization where possible.
    - VectorizeElementwisePass: Applies the VectorizeElementwise transformation to a function.

Functions:
    - vectorize_elementwise_pass: Creates an instance of the VectorizeElementwisePass.

Class VectorizeElementwise:
    Rewrites arithmetic operations to apply vectorization where possible.

    Note:
      Currently, we just vectorize the Cast operation. We can extend this to other arithmetic operations as well.

    Methods:
        visit_Arithmetic(op: Arithmetic): Visits an arithmetic operation and attempts to vectorize it if conditions
        are met.

Class VectorizeElementwisePass:
    Applies the VectorizeElementwise transformation to a function.

    Methods:
        process_func(func: Function) -> Function: Applies the VectorizeElementwise transformation to the given function.

Function vectorize_elementwise_pass:
    Creates an instance of the VectorizeElementwisePass.

    Returns:
        VectorizeElementwisePass: An instance of the VectorizeElementwisePass class.
"""
from hidet.ir.func import Function
from hidet.transforms.base import FunctionPass

from hidet.ir.tools import infer_type
from hidet.ir.functors import IRRewriter
from hidet.ir.primitives.cuda.cvt import cvtv_func
from hidet.ir.cute.ops import Arithmetic, Cast
from hidet.ir.cute.layout import TiledTensorLayout
from hidet.ir.cute.type import TiledTensorType


class VectorizeElementwise(IRRewriter):
    def visit_Arithmetic(self, op: Arithmetic):
        if isinstance(op, Cast):
            inputs = op.inputs
            x = self.visit(inputs[0])
            x_ty = infer_type(x)
            assert isinstance(x_ty, TiledTensorType)
            layout = x_ty.layout
            if isinstance(layout, TiledTensorLayout):
                size = layout.val_count()
            else:
                size = layout.count()
            attrs = op.attrs
            src_dtype = x_ty.dtype
            dst_dtype = attrs["dtype"]
            # The vectorization is quite straightforward.
            # If the vectorization is possible, we just lookup the cvtv function
            # from the registered conversion functions and replace the scalar cast
            # operation with the vectorized cast operation.
            if src_dtype.is_float() and dst_dtype.is_float():
                cvtv = cvtv_func(src_dtype, dst_dtype)
            # currently supported low-bit data types include u4, u2, u1
            # all of them satisfy 8 % bits == 0
            elif src_dtype.is_integer_subbyte():
                if size * src_dtype.nbits % 32 == 0:
                    vector_length = 32 // src_dtype.nbits
                    cvtv = cvtv_func(src_dtype, dst_dtype, vector_length)
                elif size * src_dtype.nbits % 16 == 0:
                    vector_length = 16 // src_dtype.nbits
                    cvtv = cvtv_func(src_dtype, dst_dtype, vector_length)
                elif size * src_dtype.nbits % 8 == 0:
                    vector_length = 8 / src_dtype.nbits
                    cvtv = cvtv_func(src_dtype, dst_dtype, vector_length)
                else:
                    cvtv = None
            else:
                cvtv = None
            if cvtv is not None:
                return Arithmetic([x], cvtv)
        return super().visit_Arithmetic(op)


class VectorizeElementwisePass(FunctionPass):
    def process_func(self, func: Function) -> Function:
        return self.apply_transforms(func, [VectorizeElementwise()])


def vectorize_elementwise_pass() -> FunctionPass:
    return VectorizeElementwisePass()
