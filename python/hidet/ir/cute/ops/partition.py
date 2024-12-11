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
from typing import List, Tuple

from hidet.ir.type import BaseType
from hidet.ir.expr import Expr, is_constant

from hidet.ir.cute.expr import Op
from hidet.ir.cute.algorithm import TiledCopy, is_auto_copy, TiledMma, is_auto_mma
from hidet.ir.cute.type import tiled_tensor, TiledTensorType, logical_encoding
from hidet.ir.cute import (
    TensorLayout,
    TiledTensorLayout,
    composition,
    make_layout,
    auto_layout,
    logical_divide,
    product_each,
    prefix_product,
    is_integer,
    filter_zeros,
)


def reg_tensor_stride(a, b):
    if isinstance(a, tuple):
        assert isinstance(b, tuple) and len(a) == len(b)
        return tuple(reg_tensor_stride(x, y) for x, y in zip(a, b))
    else:
        assert is_integer(b)
        return 0 if is_constant(b) and b == 0 else a


def infer_type(x_type: BaseType, shape: Tuple[int], thrval_layout: TensorLayout):
    if x_type.scope.is_register():
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

    layouts = [thrval_layout[0][1], thrval_layout[1]]
    x_shape = product_each(x_type.layout.shape)
    if x_shape != shape:
        if (
            len(x_shape) < len(shape)
            or any(a != b for a, b in zip(x_shape[: len(shape) - 1], shape[:-1]))
            or x_shape[len(shape) - 1] < shape[-1]
        ):
            raise ValueError(f"Invalid argument.(got:x({x_shape}),tiled_copy({shape}))")
        rest = logical_divide(TensorLayout(x_shape), TensorLayout(shape))[1:]
        from hidet.ir.cute.int_tuple import size

        remain = x_shape[len(shape) :]
        remain_dim = rest.size() // size(remain)
        remain = (remain_dim,) + remain if remain_dim > 1 else remain
        rest = composition(rest, TensorLayout(remain))
        for i in rest:
            layouts.append(i)
    return tiled_tensor(x_type.dtype, x_type.layout.compose(make_layout(*layouts)), x_type.scope)


class Partition(Op):
    def resolve_logical_encoding(self):
        if is_auto_copy(self.tiled_copy):
            raise RuntimeError(
                "Cannot resolve the logical encoding for tensors because the "
                f"tiled_copy hasn't been specified.(got:{self.tiled_copy.str_indented()})"
            )
        shape, src_tv = self.tiled_copy.src_tv_layout()
        _, dst_tv = self.tiled_copy.dst_tv_layout()
        return logical_encoding(shape, src_tv), logical_encoding(shape, dst_tv)


class PartitionSrc(Partition):
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
        if is_auto_copy(self.tiled_copy):
            return tiled_tensor(x_type.dtype, auto_layout, x_type.scope)
        shape, thrval_layout = self.tiled_copy.src_tv_layout()
        return infer_type(x_type, shape, thrval_layout)


def partition_src(x: Expr, tiled_copy: TiledCopy):
    return PartitionSrc(x, tiled_copy).make_call()


class PartitionDst(Partition):
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
        if is_auto_copy(self.tiled_copy):
            return tiled_tensor(x_type.dtype, auto_layout, x_type.scope)
        shape, thrval_layout = self.tiled_copy.dst_tv_layout()
        return infer_type(x_type, shape, thrval_layout)


def partition_dst(x: Expr, tiled_copy: TiledCopy):
    return PartitionDst(x, tiled_copy).make_call()


class PartitionA(Partition):
    """
    Partition a tensor 'x' based on the given TiledMma configuration.

    TiledMma is the task mapping for a GEMM (mma) operation. This operator partitions the
    input tensor according to the thread-value layout of the A matrix in the TiledMma configuration.

    Example:
        x = tensor(float16, (16, 32), "register")
        a_tv_layout = ((4, 8), (2, 2, 2)):((32, 1), (16, 8, 128))  # a 16x16 K-major tensor corresponding to A
        b_tv_layout = ((4, 8), (2, 2)):((16, 1), (8, 64))  # a 8x16 K-major tensor corresponding to B
        c_tv_layout = ...
        tiled_mma = TiledMma(a_tv_layout, b_tv_layout, c_tv_layout)
        partitioned_x = partition_A(x, tiled_mma)  # partition x based on the thread-value layout of A

    First, the x tensor is split into two parts:
        x1 = tensor(float16, (16, 16, 2), "register")
        The corresponding thread-value layout is
        x1_tv_layout = ((4, 8), ((2, 2, 2), 2)):((32, 1), ((16, 8, 128), 256))

    Attributes:
        x (Expr): The input tensor expression to be partitioned.
        tiled_mma (TiledMma): The TiledMma configuration used for partitioning.

    Methods:
        infer_type(arg_types: List[BaseType]) -> BaseType:
            Infers the type of the partitioned tensor based on the input types and TiledMma configuration.
    """

    def __init__(self, x: Expr, tiled_mma: TiledMma):
        """
        Initializes the PartitionA instance.

        Args:
            x (Expr): The input tensor expression to be partitioned.
            tiled_mma (TiledMma): The TiledMma configuration used for partitioning.
        """
        super().__init__(args=[x], attrs={"tiled_mma": tiled_mma})
        self.x: Expr = x
        self.tiled_mma: TiledMma = tiled_mma

    def infer_type(self, arg_types: List[BaseType]) -> BaseType:
        """
        Infers the type of the partitioned tensor based on the input types and TiledMma configuration.

        Args:
            arg_types (List[BaseType]): The list of input types for the tensor.

        Returns:
            BaseType: The inferred type of the partitioned tensor.

        Raises:
            TypeError: If the input type is not a TiledTensorType.
        """

        x_type = arg_types[0]
        if not isinstance(x_type, TiledTensorType):
            raise TypeError(f"Type mismatch.(got:x({x_type}))")
        if x_type.layout is auto_layout:
            return tiled_tensor(x_type.dtype, auto_layout, x_type.scope)
        if is_auto_mma(self.tiled_mma):
            return tiled_tensor(x_type.dtype, auto_layout, x_type.scope)
        shape, thrval_layout = self.tiled_mma.a_tv_layout()
        return infer_type(x_type, shape, thrval_layout)


def partition_A(x: Expr, tiled_mma: TiledMma):
    return PartitionA(x, tiled_mma).make_call()


class PartitionB(Partition):
    """
    Partition a tensor 'x' based on the given TiledMma configuration.

    TiledMma is the task mapping for a GEMM (mma) operation. This operator partitions the
    input tensor according to the thread-value layout of the B matrix in the TiledMma configuration.
    The semantics of this operator are similar to PartitionA.

    Attributes:
        x (Expr): The input tensor expression to be partitioned.
        tiled_mma (TiledMma): The TiledMma configuration used for partitioning.

    Methods:
        infer_type(arg_types: List[BaseType]) -> BaseType:
            Infers the type of the partitioned tensor based on the input types and TiledMma configuration.
    """

    def __init__(self, x: Expr, tiled_mma: TiledMma):
        """
        Initializes the PartitionB instance.

        Args:
            x (Expr): The input tensor expression to be partitioned.
            tiled_mma (TiledMma): The TiledMma configuration used for partitioning.
        """
        super().__init__(args=[x], attrs={"tiled_mma": tiled_mma})
        self.x: Expr = x
        self.tiled_mma: TiledMma = tiled_mma

    def infer_type(self, arg_types: List[BaseType]) -> BaseType:
        """
        Infers the type of the partitioned tensor based on the input types and TiledMma configuration.

        Args:
            arg_types (List[BaseType]): The list of input types for the tensor.

        Returns:
            BaseType: The inferred type of the partitioned tensor.

        Raises:
            TypeError: If the input type is not a TiledTensorType.
        """
        x_type = arg_types[0]
        if not isinstance(x_type, TiledTensorType):
            raise TypeError(f"Type mismatch.(got:x({x_type}))")
        if x_type.layout is auto_layout:
            return tiled_tensor(x_type.dtype, auto_layout, x_type.scope)
        if is_auto_mma(self.tiled_mma):
            return tiled_tensor(x_type.dtype, auto_layout, x_type.scope)
        shape, thrval_layout = self.tiled_mma.b_tv_layout()
        return infer_type(x_type, shape, thrval_layout)


def partition_B(x: Expr, tiled_mma: TiledMma):
    return PartitionB(x, tiled_mma).make_call()
