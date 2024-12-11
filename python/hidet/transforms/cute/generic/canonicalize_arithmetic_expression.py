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
from typing import Type, Dict, Union

from hidet.ir.type import DataType
from hidet.ir import expr
from hidet.ir.expr import Expr, UnaryExpr, BinaryExpr
from hidet.ir.tools import TypeInfer
from hidet.ir.functors import IRRewriter

from hidet.ir.cute.type import TiledTensorType
from hidet.ir.func import Function
from hidet.transforms.base import FunctionPass

from hidet.ir.cute.ops import arithmetic
from hidet.ir.cute.ops.arithmetic import UnaryOp, BinaryOp, Neg, Add, Sub, Multiply, Div


_convert_table: Dict[Type[Expr], Type[Union[UnaryOp, BinaryOp]]] = {
    # unary
    expr.Neg: Neg,
    # binary
    expr.Add: Add,
    expr.Sub: Sub,
    expr.Multiply: Multiply,
    expr.Div: Div,
}


class CanonicalizeArithmeticExpression(IRRewriter):
    """
    A class to canonicalize arithmetic expressions.

    In the design of the Cute IR, there are two ways to create an elementwise operation.

    First, we support some sytax sugar for arithmetic expressions.
    For example, we can write `a + b` instead of `arithmetic(a, b, op=add)`.

    Second, we support creating an elementwise operation with the underlying elementwise
    function.
    For example, we can write `arithmetic(a, b, op=add)` to create an elementwise addition.
    This allows user to define their own elementwise operations.
    This is useful when we want to fuse consecutive elementwise operations. Consider the following example:
    ```python
    a = make_tensor("float16", [128, 128], "register")
    b = make_tensor("float16", [128, 128], "register")
    c = a + b
    d = relu(c)
    e = d * 2
    ```
    We can design a pass to fuse the elementwise operations `c = a + b`, `d = relu(c)`, and `e = d * 2` into
    a single elementwise operation. The pass will replace operator `d * 2` with
    ```python
    def elementwise_add_relu_multiply(a, b):
        return relu(a + b) * 2
    e = arithmetic(a, b, op=elementwise_add_relu_multiply)
    ```
    Then, the above IR will be transformed into a single nested loop:
    ```python
    for coords in grid(local_shape):
        c[cords] = relu(a[coords] + b[coords]) * 2
    ```

    This pass basically canonicalizes the arithmetic expressions to the second form.

    Attributes:
        type_infer (TypeInfer): An instance of TypeInfer to infer types of expressions.

    Methods:
        visit_Unary(e: UnaryExpr) -> Expr:
            Processes and canonicalizes unary expressions.

        visit_Binary(e: BinaryExpr) -> Expr:
            Processes and canonicalizes binary expressions.
    """

    def __init__(self):
        """Initializes a CanonicalizeArithmeticExpression object."""
        super().__init__()
        self.type_infer = TypeInfer()

    def visit_Unary(self, e: UnaryExpr):
        """
        Processes and canonicalizes unary expressions.

        Args:
            e (UnaryExpr): The unary expression to process.

        Returns:
            Expr: The canonicalized expression.
        """
        op_cls = type(e)
        a = self.visit(e.a)
        a_type = self.type_infer(a)
        if isinstance(a_type, TiledTensorType):
            if op_cls not in _convert_table:
                raise NotImplementedError(f"{op_cls} not implemented now")
            cute_op_cls = _convert_table[op_cls]
            return cute_op_cls(a).make_call()
        else:
            return super().visit_Unary(e)

    def visit_Binary(self, e: BinaryExpr):
        """
        Processes and canonicalizes binary expressions.

        Args:
            e (BinaryExpr): The binary expression to process.

        Returns:
            Expr: The canonicalized expression.
        """
        op_cls = type(e)
        a = self.visit(e.a)
        b = self.visit(e.b)
        a_type = self.type_infer(a)
        b_type = self.type_infer(b)
        if isinstance(a_type, TiledTensorType):
            if isinstance(b_type, TiledTensorType):
                if op_cls not in _convert_table:
                    raise NotImplementedError(f"{op_cls} not implemented now")
                cute_op_cls = _convert_table[op_cls]
                return cute_op_cls(a, b).make_call()
            elif isinstance(b_type, DataType):

                def binary(a):
                    return Expr._binary(op_cls, a, b)  # pylint: disable=protected-access

                binary.__name__ = binary.__name__ + f"_{op_cls}"
                return arithmetic(a, op=binary)
            else:
                raise TypeError(
                    f"{op_cls} should have TiledTensorType or DataType operand(got:a({a_type}),b({b_type}))"
                )
        elif isinstance(b_type, TiledTensorType):
            if isinstance(a_type, DataType):

                def binary(b):
                    return Expr._binary(op_cls, a, b)  # pylint: disable=protected-access

                binary.__name__ = binary.__name__ + f"_{op_cls}"
                return arithmetic(b, op=binary)
            else:
                raise TypeError(
                    f"{op_cls} should have TiledTensorType or DataType operand(got:a({a_type}),b({b_type}))"
                )
        else:
            return super().visit_Binary(e)


class CanonicalizeArithmeticExpressionPass(FunctionPass):
    def process_func(self, func: Function) -> Function:
        return self.apply_transforms(func, [CanonicalizeArithmeticExpression()])


def canonicalize_arithmetic_expression_pass() -> FunctionPass:
    return CanonicalizeArithmeticExpressionPass()
