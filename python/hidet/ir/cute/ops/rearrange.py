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
from typing import List, Union

from hidet.ir.type import BaseType
from hidet.ir.expr import Expr
from hidet.ir.stmt import DeclareScope

from hidet.ir.cute.expr import Op
from hidet.ir.cute.type import tiled_tensor, TiledTensorType
from hidet.ir.cute import TensorLayout
from hidet.ir.cute.layout import TiledTensorLayout


class Rearrange(Op):
    """
    Rearranges the data layout of a tensor within the register scope. This operator redistributes
    the data in a tensor according to the specified layout while ensuring the tensor resides in register files.

    Attributes:
        x (Expr): The tensor expression to be rearranged.
        layout (Union[TiledTensorLayout, TensorLayout]): The target layout for the tensor.
        scope (DeclareScope): The scope within which the tensor resides, must be a register scope.

    Methods:
        infer_type(arg_types: List[BaseType]) -> BaseType:
            Infers the type of the result of the rearrange operation based on input types.
    """

    def __init__(self, x: Expr, layout: Union[TiledTensorLayout, TensorLayout], scope: Union[DeclareScope, str]):
        """
        Initializes the Rearrange class with the given tensor, layout, and scope.

        Args:
            x (Expr): The tensor expression to be rearranged.
            layout (Union[TiledTensorLayout, TensorLayout]): The target layout for the tensor.
            scope (Union[DeclareScope, str]): The scope within which the tensor resides. Must be a register scope.
        """

        if isinstance(scope, str):
            scope = DeclareScope.from_str(scope)
        assert scope.is_register(), "The rearrange only performs redistribution of data in register files"
        super().__init__(args=[x], attrs={"layout": layout, "scope": scope})
        self.x: Expr = x
        self.layout = layout
        self.scope: DeclareScope = scope

    def infer_type(self, arg_types: List[BaseType]) -> BaseType:
        """
        Infers the type of the result of the rearrange operation based on input types.

        Args:
            arg_types (List[BaseType]): List of input types.

        Returns:
            BaseType: The inferred type of the result.
        """

        x_type = arg_types[0]
        assert isinstance(x_type, TiledTensorType)
        assert isinstance(x_type.layout, TiledTensorLayout) and isinstance(self.layout, TiledTensorLayout)
        assert x_type.layout.shape() == self.layout.shape()
        return tiled_tensor(x_type.dtype, self.layout, self.scope)


def rearrange(x: Expr, layout: Union[TiledTensorLayout, TensorLayout], scope: Union[DeclareScope, str]):
    return Rearrange(x, layout, scope).make_call()
