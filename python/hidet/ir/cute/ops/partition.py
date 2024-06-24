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
from typing import List

from hidet.ir.type import BaseType
from hidet.ir.expr import Expr, is_constant
from hidet.ir.stmt import DeclareScope

from hidet.ir.cute.expr import Op
from hidet.ir.cute.algorithm import TiledCopy
from hidet.ir.cute.type import tiled_tensor, TiledTensorType
from hidet.ir.cute import TensorLayout, TiledTensorLayout, make_layout, prefix_product, is_integer, filter_zeros


def reg_tensor_stride(a, b):
    if isinstance(a, tuple):
        assert isinstance(b, tuple) and len(a) == len(b)
        return tuple(reg_tensor_stride(x, y) for x, y in zip(a, b))
    else:
        assert is_integer(b)
        return 0 if is_constant(b) and b == 0 else a


class PartitionSrc(Op):
    """
    Partition a tensor for a copy operation. This operator partitions the
    source tensor into a local tensor held by each thread. During code generation,
    this operation pre-calculates the offsets associated with the thread index,
    allowing loop-invariant computation to be moved out of the copy loop.

    Example:
    Assume we have a copy operation defined by the following TV layouts:
    ```python
    src_tv_layout = TensorLayout(((4, 8), (2, 2)), ((32, 1), (16, 8)))
    dst_tv_layout = src_tv_layout
    ```
    Each thread holds a local tensor of size (2, 2). For a specific thread with index `tid`,
    the offset of the local tensor is calculated as:
    ```python
    offset = (tid % 4) * 32 + ((tid // 4) % 8) * 1
    ```
    This offset is moved out of the copy loop to reduce the overhead of the copy operation.

    Attributes:
        x (Expr): The source tensor expression.
        tiled_copy (TiledCopy): The tiled copy operation associated with this partition.

    Methods:
        infer_type(arg_types: List[BaseType]) -> BaseType:
            Infers the type of the result of the partition operation based on input types.
    """

    def __init__(self, x: Expr, tiled_copy: TiledCopy):
        """
        Initializes the PartitionSrc class with the given source tensor and tiled copy operation.

        Args:
            x (Expr): The source tensor expression.
            tiled_copy (TiledCopy): The tiled copy operation associated with this partition.
        """

        super().__init__(args=[x], attrs={"tiled_copy": tiled_copy})
        self.x: Expr = x
        self.tiled_copy: TiledCopy = tiled_copy

    def infer_type(self, arg_types: List[BaseType]) -> BaseType:
        """
        Infers the type of the result of the partition operation based on input types.

        Args:
            arg_types (List[BaseType]): List of input types.

        Returns:
            BaseType: The inferred type of the result.
        """

        x_type = arg_types[0]
        if not isinstance(x_type, TiledTensorType):
            raise TypeError(f"Type mismatch.(got:x({x_type}))")
        shape, thrval_layout = self.tiled_copy.src_tv_layout()
        if x_type.scope == DeclareScope.Register:
            shape = (thrval_layout[0][1].shape, thrval_layout[1].shape)
            stride = (thrval_layout[0][1].stride, thrval_layout[1].stride)
            shape_fz = filter_zeros(stride, shape)
            stride = reg_tensor_stride(prefix_product(shape_fz), stride)
            layout = TensorLayout(shape, stride)
            if not isinstance(x_type.layout, TiledTensorLayout):
                raise TypeError(
                    "Invalid layout for partitioning register tensor. "
                    f"got(x({x_type.layout}),expected(TiledTensorLayout))"
                )
            x_layout = x_type.layout
            if x_layout.val_count() == layout.count():
                return tiled_tensor(x_type.dtype, layout, x_type.scope)
            else:
                x_cnt = x_layout.val_count()
                cnt = layout.count()
                if x_cnt % cnt != 0:
                    raise TypeError(
                        "Can't partition the register tensor due to divisibility. "
                        f"got(x({x_layout.val_layout()}),expected({layout}))"
                    )
                shape = layout.shape + (x_cnt // cnt,)
                stride = layout.stride + (cnt,)
                return tiled_tensor(x_type.dtype, TensorLayout(shape, stride), x_type.scope)

        return tiled_tensor(
            x_type.dtype, x_type.layout.compose(make_layout(thrval_layout[0][1], thrval_layout[1])), x_type.scope
        )


def partition_src(x: Expr, tiled_copy: TiledCopy):
    return PartitionSrc(x, tiled_copy).make_call()


class PartitionDst(Op):
    """
    Partition a tensor for a copy operation. This operator partitions the
    destination tensor into a local tensor held by each thread. During code generation,
    this operation pre-calculates the offsets associated with the thread index,
    allowing loop-invariant computation to be moved out of the copy loop.

    Attributes:
        x (Expr): The destination tensor expression.
        tiled_copy (TiledCopy): The tiled copy operation associated with this partition.

    Methods:
        infer_type(arg_types: List[BaseType]) -> BaseType:
            Infers the type of the result of the partition operation based on input types.
    """

    def __init__(self, x: Expr, tiled_copy: TiledCopy):
        """
        Initializes the PartitionDst class with the given destination tensor and tiled copy operation.

        Args:
            x (Expr): The destination tensor expression.
            tiled_copy (TiledCopy): The tiled copy operation associated with this partition.
        """

        super().__init__(args=[x], attrs={"tiled_copy": tiled_copy})
        self.x: Expr = x
        self.tiled_copy: TiledCopy = tiled_copy

    def infer_type(self, arg_types: List[BaseType]) -> BaseType:
        """
        Infers the type of the result of the partition operation based on input types.

        Args:
            arg_types (List[BaseType]): List of input types.

        Returns:
            BaseType: The inferred type of the result.
        """

        x_type = arg_types[0]
        if not isinstance(x_type, TiledTensorType):
            raise TypeError(f"Type mismatch.(got:x({x_type}))")
        shape, thrval_layout = self.tiled_copy.dst_tv_layout()
        if x_type.scope == DeclareScope.Register:
            shape = (thrval_layout[0][1].shape, thrval_layout[1].shape)
            stride = (thrval_layout[0][1].stride, thrval_layout[1].stride)
            shape_fz = filter_zeros(stride, shape)
            stride = reg_tensor_stride(prefix_product(shape_fz), stride)
            layout = TensorLayout(shape, stride)
            if not isinstance(x_type.layout, TiledTensorLayout):
                raise TypeError(
                    "Invalid layout for partitioning register tensor. "
                    f"got(x({x_type.layout}),expected(TiledTensorLayout))"
                )
            x_layout = x_type.layout
            if x_layout.val_count() == layout.count():
                return tiled_tensor(x_type.dtype, layout, x_type.scope)
            else:
                x_cnt = x_layout.val_count()
                cnt = layout.count()
                if x_cnt % cnt != 0:
                    raise TypeError(
                        "Can't partition the register tensor due to divisibility. "
                        f"got(x({x_layout.val_layout()}),expected({layout}))"
                    )
                shape = layout.shape + (x_cnt // cnt,)
                stride = layout.stride + (cnt,)
                return tiled_tensor(x_type.dtype, TensorLayout(shape, stride), x_type.scope)

        return tiled_tensor(
            x_type.dtype, x_type.layout.compose(make_layout(thrval_layout[0][1], thrval_layout[1])), x_type.scope
        )


def partition_dst(x: Expr, tiled_copy: TiledCopy):
    return PartitionDst(x, tiled_copy).make_call()
