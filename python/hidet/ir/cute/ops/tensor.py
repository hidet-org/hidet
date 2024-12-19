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
from typing import List, Union, Optional
from abc import abstractmethod

from hidet.ir.cute.layout import TiledTensorLayout, ComposedTensorLayout, TensorLayout, is_auto_layout
from hidet.ir.expr import Expr
from hidet.ir.cute.expr import Op
from hidet.ir.cute.type import tiled_tensor, TiledTensorType
from hidet.ir.type import BaseType, DataType, PointerType, TensorType, TensorPointerType, data_type

from hidet.ir.stmt import DeclareScope


class TensorBase:
    """
    An abstract base class for tensor operations, defining the interface for checking volatility.
    """

    @abstractmethod
    def is_volatile(self):
        """
        Check if the tensor is volatile. If true, the compiler won't perform bank conflict elimination on this tensor.
        Returns:
            bool: True if the tensor is volatile, False otherwise.
        """

        raise NotImplementedError


class Tensor(Op, TensorBase):
    """
    Creates a tensor with a specific data type, layout, and scope.
    Note that the compiler will manage the memory allocation for the tensor, so the tensor cannot
    be created in the global memory.

    Attributes:
        dtype (DataType): The data type of the tensor.
        layout (Union[TiledTensorLayout, TensorLayout]): The layout of the tensor.
        scope (DeclareScope): The scope in which the tensor resides.

    Methods:
        is_volatile() -> bool:
            Check if the tensor is volatile.

        infer_type(arg_types: List[BaseType]) -> BaseType:
            Infers the type of the tensor based on input argument types.
    """

    def __init__(
        self,
        dtype: Union[DataType, str],
        layout: Union[TiledTensorLayout, TensorLayout],
        scope: Union[DeclareScope, str] = None,
    ):
        """
        Initializes the Tensor with the given data type, layout, and scope.

        Args:
            dtype (Union[DataType, str]): The data type of the tensor.
            layout (Union[TiledTensorLayout, TensorLayout]): The layout of the tensor.
            scope (Union[DeclareScope, str]): The scope in which the tensor resides.
        """
        if isinstance(dtype, str):
            dtype = data_type(dtype)
        if isinstance(scope, str):
            scope = DeclareScope.from_str(scope)
        super().__init__(args=[], attrs={"dtype": dtype, "layout": layout, "scope": scope})
        assert (
            (isinstance(layout, TiledTensorLayout) and scope.is_register())
            or isinstance(layout, (TensorLayout, ComposedTensorLayout))
            or is_auto_layout(layout)
        )
        # cannot create a global tensor
        if scope.is_global():
            raise ValueError("cannot create a tenosor in the global memory, please use TensorView instead.")
        if not isinstance(dtype, DataType):
            raise TypeError(f"invalid data type.(got:{dtype})")
        self.dtype = dtype
        self.layout = layout
        self.scope = scope

    def is_volatile(self):
        """
        Check if the tensor is volatile.

        Returns:
            bool: Always returns False, indicating the tensor is not volatile.
        """
        return False

    def infer_type(self, arg_types: List[BaseType]) -> BaseType:
        """
        Infers the type of the tensor based on the types of its arguments.

        Args:
            arg_types (List[BaseType]): List of argument types.

        Returns:
            BaseType: The inferred type of the tensor.
        """
        return tiled_tensor(dtype=self.dtype, layout=self.layout, scope=self.scope)


def make_tensor(
    dtype: Union[DataType, str],
    layout: Union[TiledTensorLayout, ComposedTensorLayout, TensorLayout],
    scope: Union[DeclareScope, str] = None,
):
    return Tensor(dtype, layout, scope).make_call()


class TensorView(Op, TensorBase):
    """
    View a tensor within global/shared memory or register files with a specified layout and scope.

    This operator creates a new view of the tensor data, possibly facilitating further optimizations
    in the CuTE dialect. The tensor data is not copied, but the view is created with the specified layout
    and scope.

    Attributes:
        x (Expr): The tensor expression to be viewed.
        layout (Union[TiledTensorLayout, TensorLayout]): The target layout for the tensor.
        scope (Union[DeclareScope, str]): The scope within which the tensor resides.
        volatile (bool): Indicates if the tensor view is volatile.

    Methods:
        is_volatile() -> bool:
            Check if the tensor view is volatile.

        infer_type(arg_types: List[BaseType]) -> BaseType:
            Infers the type of the result of the tiled tensor view operation based on input types.
    """

    def __init__(
        self,
        x: Expr,
        layout: Union[TiledTensorLayout, TensorLayout],
        scope: Union[DeclareScope, str] = None,
        volatile: Optional[bool] = False,
    ):
        """
        Initializes the TensorView with the given tensor, layout, scope, and volatility.

        Args:
            x (Expr): The tensor expression to be viewed.
            layout (Union[TiledTensorLayout, TensorLayout]): The target layout for the tensor.
            scope (Union[DeclareScope, str]): The scope within which the tensor resides.
            volatile (Optional[bool]): Indicates if the tensor view is volatile. Defaults to False.
        """
        if isinstance(scope, str):
            scope = DeclareScope.from_str(scope)
        super().__init__(args=[x], attrs={"layout": layout, "scope": scope, "volatile": volatile})
        self.x: Expr = x
        assert (
            (isinstance(layout, TiledTensorLayout) and scope.is_register())
            or isinstance(layout, (TensorLayout, ComposedTensorLayout))
            or is_auto_layout(layout)
        )
        self.layout = layout
        self.scope = scope
        self.volatile = volatile

    def is_volatile(self):
        """
        Check if the tensor view is volatile.

        Returns:
            bool: The volatility status of the tensor view.
        """
        return self.volatile

    def infer_type(self, arg_types: List[BaseType]) -> BaseType:
        """
        Infers the type of the tensor view based on the types of its arguments.

        Args:
            arg_types (List[BaseType]): List of argument types.

        Returns:
            BaseType: The inferred type of the tensor view.

        Raises:
            AssertionError: If the argument types are not as expected.
        """
        x_type = arg_types[0]
        assert isinstance(x_type, (TiledTensorType, TensorType, TensorPointerType, PointerType))
        import math

        if isinstance(x_type, TiledTensorType):
            dtype = x_type.dtype
            if is_auto_layout(x_type.layout):
                tensor_size = None
            else:
                assert isinstance(x_type.layout, (TensorLayout, ComposedTensorLayout))
                tensor_size = x_type.layout.count()
        elif isinstance(x_type, TensorPointerType):
            ttype = x_type.tensor_type
            dtype = ttype.dtype
            tensor_size = None
        elif isinstance(x_type, TensorType):
            dtype = x_type.dtype
            tensor_size = math.prod(x_type.shape)
        else:
            dtype = x_type.base_type
            if isinstance(dtype, TensorPointerType):
                dtype = dtype.tensor_type.dtype
            tensor_size = None
        if is_auto_layout(self.layout):
            pass
        elif isinstance(self.layout, TiledTensorLayout):
            assert tensor_size is None or tensor_size == self.layout.val_count()
        else:
            assert isinstance(self.layout, (TensorLayout, ComposedTensorLayout))
            assert tensor_size is None or tensor_size == self.layout.size()
        return tiled_tensor(dtype=dtype, layout=self.layout, scope=self.scope)


def tensor_view(
    x: Expr,
    layout: Union[TiledTensorLayout, ComposedTensorLayout, TensorLayout],
    scope: Union[DeclareScope, str] = None,
    volatile: Optional[bool] = False,
):
    return TensorView(x, layout, scope, volatile).make_call()
