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

from hidet.ir.cute.layout import TiledTensorLayout, ComposedTensorLayout, TensorLayout
from hidet.ir.expr import Expr
from hidet.ir.cute.expr import Op
from hidet.ir.cute.type import tiled_tensor, TiledTensorType
from hidet.ir.type import BaseType, PointerType, TensorType, TensorPointerType

from hidet.ir.stmt import DeclareScope


class TiledTensorView(Op):
    """
    View a tensor within global/shared memory or register files with a specified layout and scope.
    This operator creates a new view of the tensor data, possibly facilitating further optimizations
    in the CuTE dialect. The tensor data is not copied, but the view is created with the specified layout
    and scope.

    Attributes:
        x (Expr): The tensor expression to be viewed.
        layout (Union[TiledTensorLayout, TensorLayout]): The target layout for the tensor.
        scope (Union[DeclareScope, str]): The scope within which the tensor resides.

    Methods:
        infer_type(arg_types: List[BaseType]) -> BaseType:
            Infers the type of the result of the tiled tensor view operation based on input types.
    """

    def __init__(self, x: Expr, layout: Union[TiledTensorLayout, TensorLayout], scope: Union[DeclareScope, str] = None):
        """
        Initializes the TiledTensorView class with the given tensor, layout, and optional scope.

        Args:
            x (Expr): The tensor expression to be viewed.
            layout (Union[TiledTensorLayout, TensorLayout]): The target layout for the tensor.
            scope (Union[DeclareScope, str], optional): The scope within which the tensor resides. Defaults to None.
        """

        if isinstance(scope, str):
            scope = DeclareScope.from_str(scope)
        super().__init__(args=[x], attrs={"layout": layout, "scope": scope})
        self.x: Expr = x
        assert (isinstance(layout, TiledTensorLayout) and scope == DeclareScope.Register) or isinstance(
            layout, (TensorLayout, ComposedTensorLayout)
        )
        self.layout = layout
        self.scope = scope

    def infer_type(self, arg_types: List[BaseType]) -> BaseType:
        """
        Infers the type of the result of the tiled tensor view operation based on input types.

        Args:
            arg_types (List[BaseType]): List of input types.

        Returns:
            BaseType: The inferred type of the result.
        """

        x_type = arg_types[0]
        assert isinstance(x_type, (TiledTensorType, TensorType, TensorPointerType, PointerType))
        import math

        if isinstance(x_type, TiledTensorType):
            dtype = x_type.dtype
            assert isinstance(x_type.layout, (TensorLayout, ComposedTensorLayout))
            tensor_size = x_type.layout.count()
        elif isinstance(x_type, TensorPointerType):
            ttype = x_type.tensor_type
            dtype = ttype.dtype
            tensor_size = None
        elif isinstance(x_type, TensorType):
            ttype = x_type
            dtype = ttype.dtype
            tensor_size = math.prod(ttype.shape)
        else:
            dtype = x_type.base_type
            if isinstance(dtype, TensorPointerType):
                dtype = dtype.tensor_type.dtype
            tensor_size = None
        if isinstance(self.layout, TiledTensorLayout):
            assert tensor_size is None or tensor_size == self.layout.val_count()
        else:
            assert isinstance(self.layout, (TensorLayout, ComposedTensorLayout))
            assert tensor_size is None or tensor_size == self.layout.size()
        return tiled_tensor(dtype=dtype, layout=self.layout, scope=self.scope)


def tiled_tensor_view(x: Expr, layout: Union[TiledTensorLayout, TensorLayout], scope: Union[DeclareScope, str] = None):
    return TiledTensorView(x, layout, scope).make_call()
