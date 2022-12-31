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
Lower special cast pass
=======================

By default, we reply on the underlying C cast to cast an expression from one data type to another. like

.. code-block:: c

    int64_t a = 128;
    float b = (float)a;

But, some data types are not supported by the underlying C cast. For example, we cannot cast an int64_t number to
a float16 number. In this case, we need to use a special cast function to cast the expression. Hidet allows developers
to implement the special cast function in hidet.ir.primitives.math.MathFunctionSet.cast(). This pass will lower the
cast expression to the special cast function defined in the MathFunctionSet.

For example, we use cvt function to implement the cast from int64_t to float16. And this pass will lower the program:

.. code-block:: c

    int64_t a = 128;
    float16_t b = (float16_t)a;

to

.. code-block:: c

    int64_t a = 128;
    float16_t b = cuda_cvt_int64_to_float16(a);

"""
from typing import Optional
from hidet.ir.type import DataType
from hidet.ir.stmt import Stmt
from hidet.ir.expr import Cast, Expr
from hidet.ir.func import Function
from hidet.ir.functors import StmtExprRewriter, TypeInfer
from hidet.transforms.base import FunctionBodyPass
from hidet.ir.primitives.math import MathFunctionSet, registered_math_function_sets


class LowerCastRewriter(StmtExprRewriter):
    def __init__(self, device: str):
        super().__init__()
        self.device: str = device
        self.type_infer: TypeInfer = TypeInfer()

    def visit_Cast(self, e: Cast):
        target_type = e.target_type
        if isinstance(target_type, DataType):
            src_expr: Expr = self(e.expr)
            src_dtype = self.type_infer(src_expr)
            if not isinstance(src_dtype, DataType):
                # convert from non-dtype (e.g., pointer) to dtype
                return StmtExprRewriter.visit_Cast(self, e)
            if (self.device, src_dtype.name) in registered_math_function_sets:
                function_set: MathFunctionSet = registered_math_function_sets[(self.device, src_dtype.name)]
                casted: Optional[Expr] = function_set.cast(src_expr, target_type)
            else:
                # use default cast
                casted = None
            if casted is None:
                return StmtExprRewriter.visit_Cast(self, e)
            else:
                return casted

        return StmtExprRewriter.visit_Cast(self, e)


class LowerSpecialCastPass(FunctionBodyPass):
    def __init__(self):
        super().__init__()
        self.device: Optional[str] = None

    def process_func(self, func: Function) -> Function:
        func_kind_to_device = {'host_kernel': 'cpu', 'packed_func': 'cpu', 'cuda_kernel': 'cuda', 'cuda_device': 'cuda'}
        self.device = func_kind_to_device[func.kind]
        return FunctionBodyPass.process_func(self, func)

    def process_body(self, stmt: Stmt) -> Stmt:
        rewriter = LowerCastRewriter(self.device)
        return rewriter.rewrite(stmt)


def lower_special_cast_pass():
    return LowerSpecialCastPass()
