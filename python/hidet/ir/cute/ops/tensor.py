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
from typing import Tuple, List, Union, Optional
from abc import abstractmethod

from hidet.ir.cute.int_tuple import product_each, Int, flatten
from hidet.ir.cute.layout import (
    LayoutBase,
    TiledTensorLayout,
    ComposedTensorLayout,
    TensorLayout,
    is_auto_layout,
    make_layout,
)
from hidet.ir.expr import Expr, is_constant
from hidet.ir.cute.expr import Op
from hidet.ir.cute.type import tiled_tensor, TiledTensorType
from hidet.ir.type import BaseType, DataType, PointerType, TensorType, TensorPointerType, data_type

from hidet.ir.stmt import DeclareScope


class TensorBase(Op):
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

    def resolve_logical_encoding(self):
        """
        Resolves the logical encoding for tensor layouts in the Hexcute system.

        In the current design of Hexcute, layouts are handled differently based on the memory scope:

        - For register tensors: Layouts are typically inferred by the compiler, but users can also
          explicitly annotate them. This function creates a logical encoding based on the tensor's
          shape and layout configuration.

        - For global tensors: Layouts must be explicitly specified by users since the kernel compiler
          cannot modify the layout of global tensors. The layout information is preserved as-is.

        - For shared memory tensors: Layouts are handled through a different mechanism specific to
          shared memory optimization.

        Returns:
            List[Optional[Expr]]: A list containing either:
                - A logical encoding expression for register tensors
                - None for non-register tensors (global/shared memory)
        """
        from hidet.ir.cute.type import logical_encoding

        if self.scope.is_register():
            assert not is_auto_layout(self.layout) and isinstance(self.layout, TiledTensorLayout)
            shape = self.layout.shape()
            thr, val = self.layout.thr_layout(), self.layout.val_layout()
            return [logical_encoding(shape, make_layout(thr, val))]
        else:
            return [None]


class Tensor(TensorBase):
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

    def __init__(self, dtype: Union[DataType, str], layout: LayoutBase, scope: Union[DeclareScope, str]):
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


class TensorView(TensorBase):
    """
    View a tensor within global/shared memory or register files with a specified layout and scope.

    This operator creates a new view of the tensor data, possibly facilitating further optimizations
    in the CuTE dialect. The tensor data is not copied, but the view is created with the specified layout
    and scope.

    Attributes:
        x (Expr): The tensor expression to be viewed.
        layout (Union[TiledTensorLayout, TensorLayout]): The target layout for the tensor.
        scope (Union[DeclareScope, str]): The scope within which the tensor resides.
        tile_shape (Optional[Tuple[Int, ...]]): The shape of the tile. Defaults to None. If None, the tile shape is
        equal to the shape of the layout. If the shape is explicitly specified, the tensor will be tiled.
        tile_coords (Optional[Tuple[Int, ...]]): The starting coordinates of the tile. Defaults to None. If None, the
        tile coordinates are set to 0. If the coordinates are explicitly specified, the tensor will be tiled, and the
        starting coordinates will be set accordingly.
        volatile (bool): Indicates if the tensor view is volatile.

    Methods:
        is_volatile() -> bool:
            Check if the tensor view is volatile.

        infer_type(arg_types: List[BaseType]) -> BaseType:
            Infers the type of the result of the tiled tensor view operation based on input types.


    Example: TMA copy from global memory to shared memory

    Define a tensor in global memory with row-major layout

    ```python
    g_tensor = tensor_view(
        g_pointer,
        TensorLayout((m, n), (n, 1)),  # shape: (m, n), stride: row-major
        scope="global",
        tile_shape=(bM, bN),
        tile_coords=(bidx * bM, bidx * bN)
    )

    # Allocate a tile in shared memory
    s_tensor = make_tensor(dtype, shape=(bM, bN), scope="shared")

    # Partition source and destination tensors
    g1 = partition_src(g_tensor, auto_copy())
    s1 = partition_dst(s_tensor, auto_copy())

    # Issue the copy. If TMA is applicable, the compiler will emit a TMA instruction.
    # Otherwise, it will fall back to a standard asynchronous copy.
    copy(auto_copy((bM, bN)), g1[:, :], s1[:, :], mbar=mbarrier[0])
    ```

    ------------------------------------------------------------------------------
    Notes:
    - To enable TMA:
      1. The tile_shape and tile_coords must be specified.
      2. The mbarrier must be passed to the copy operation.
      3. The compiler attempts to derive a valid shared memory layout for TMA.
         If not possible, it automatically falls back to a regular async copy.

    Design rationale:
    - On Hopper GPUs, global-to-shared memory transfers can be done using:
        1. pre-Hopper style cp.async (asynchronous copy)
        2. Hopper-specific TMA instructions

    - Both methods are valid in warp-specialized GEMM kernels.
      The choice depends on the matrix shape, leading to 4 kernel variants:
        1. TMA loads A, cp.async loads B
        2. cp.async loads A, TMA loads B
        3. TMA loads both A and B
        4. cp.async loads both A and B

    - To avoid managing all variants, we use a fallback-first design:
        > Always attempt TMA first. If TMA constraints are not met,
          automatically fall back to cp.async.

    - This simplifies maintenance: only a single GEMM kernel version is needed
      while still supporting TMA performance benefits when possible.
    """

    def __init__(
        self,
        x: Expr,
        layout: Union[TiledTensorLayout, TensorLayout],
        scope: Union[DeclareScope, str],
        tile_shape: Optional[Tuple[Int, ...]] = None,
        tile_coords: Optional[Tuple[Int, ...]] = None,
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
        super().__init__(
            args=[x],
            attrs={
                "layout": layout,
                "scope": scope,
                "tile_shape": tile_shape,
                "tile_coords": tile_coords,
                "volatile": volatile,
            },
        )
        self.x: Expr = x
        assert (
            (isinstance(layout, TiledTensorLayout) and scope.is_register())
            or isinstance(layout, (TensorLayout, ComposedTensorLayout))
            or is_auto_layout(layout)
        )
        if tile_shape is None:
            if isinstance(layout, TiledTensorLayout):
                tile_shape = layout.shape()
            elif isinstance(layout, (TensorLayout, ComposedTensorLayout)):
                tile_shape = product_each(layout.shape_tuple)
            else:
                tile_shape = None
        if tile_coords is None:
            if tile_shape is not None:
                tile_coords = [0] * len(tile_shape)
            else:
                tile_coords = None
        self.layout = layout
        self.scope = scope
        self.tile_shape = tile_shape
        self.tile_coords = tile_coords
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
            if self.scope.is_global():
                raise ValueError("The layout of a tensor in global memory should be explicitly specified.")
            layout = self.layout
        elif isinstance(self.layout, TiledTensorLayout):
            assert tensor_size is None or tensor_size == self.layout.val_count()
            layout = self.layout
        else:
            assert isinstance(self.layout, (TensorLayout, ComposedTensorLayout))
            assert tensor_size is None or tensor_size == self.layout.size()
            tile_shape = self.tile_shape
            tensor_shape = product_each(self.layout.shape_tuple)
            if tile_shape == tensor_shape:
                layout = self.layout
            else:  # tile the layout
                assert len(tile_shape) == len(tensor_shape)
                result_shape = []
                result_stride = []
                for i, extent in enumerate(tile_shape):
                    shape = flatten(self.layout[i].shape_tuple)
                    stride = flatten(self.layout[i].stride_tuple)
                    if any(not is_constant(s) for s in shape[:-1]):
                        raise TypeError(f"The dynamism of the tensor layout {self.layout} is not supported yet.")
                    cur_idx = 0
                    cur_shape = []
                    cur_stride = []
                    while extent > 1:
                        s = shape[cur_idx]
                        d = stride[cur_idx]
                        if cur_idx == len(shape) - 1 or s > extent:
                            cur_shape.append(extent)
                            cur_stride.append(d)
                            extent //= extent
                        else:
                            if extent % s != 0:
                                raise TypeError(
                                    f"The tile shape {tile_shape} is not a multiple of the tensor shape {tensor_shape}."
                                )
                            extent //= s
                            cur_shape.append(s)
                            cur_stride.append(d)
                        cur_idx += 1
                    result_shape.append(tuple(cur_shape) if len(cur_shape) > 1 else cur_shape[0])
                    result_stride.append(tuple(cur_stride) if len(cur_stride) > 1 else cur_stride[0])
                layout = TensorLayout(tuple(result_shape), tuple(result_stride))
        return tiled_tensor(dtype=dtype, layout=layout, scope=self.scope)


def tensor_view(
    x: Expr,
    layout: Union[TiledTensorLayout, ComposedTensorLayout, TensorLayout],
    scope: Union[DeclareScope, str],
    tile_shape: Optional[Tuple[Int, ...]] = None,
    tile_coords: Optional[Tuple[Int, ...]] = None,
    volatile: Optional[bool] = False,
):
    return TensorView(x, layout, scope, tile_shape=tile_shape, tile_coords=tile_coords, volatile=volatile).make_call()
