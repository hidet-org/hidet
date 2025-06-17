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
from hidet.ir.expr import Expr

from hidet.ir.cute.expr import Op
from hidet.ir.cute.algorithm import TiledCopy, is_auto_copy, TiledMma, is_auto_mma
from hidet.ir.cute.type import tiled_tensor, TiledTensorType, logical_encoding
from hidet.ir.cute.layout import is_auto_layout, canonicalize_thread_value_layout, coalesce, group, ThrValAtom
from hidet.ir.cute import (
    TensorLayout,
    TiledTensorLayout,
    composition,
    make_layout,
    auto_layout,
    logical_divide,
    product_each,
)


def validate_shape(x_shape: Tuple[int], shape: Tuple[int]):
    if x_shape != shape:
        if (
            len(x_shape) < len(shape)
            or any(a != b for a, b in zip(x_shape[: len(shape) - 1], shape[:-1]))
            or x_shape[len(shape) - 1] < shape[-1]
        ):
            raise ValueError(f"Invalid argument.(got:x({x_shape}),tiled_copy({shape}))")
        rest = logical_divide(TensorLayout(x_shape), TensorLayout(shape))[1:]
        from hidet.ir.cute.int_tuple import size

        rest_shape = x_shape[len(shape) :]
        remain_dim = rest.size() // size(rest_shape)
        rest_shape = (remain_dim,) + rest_shape if remain_dim > 1 else rest_shape
        rest_layout = composition(rest, TensorLayout(rest_shape))
        return rest_shape, rest_layout
    else:
        return [], TensorLayout(1)


def infer_type(x_type: BaseType, shape: Tuple[int], thrval_layout: TensorLayout):
    if x_type.scope.is_register():
        thrval_layout = canonicalize_thread_value_layout(thrval_layout)
        thr_layout, val_layout = thrval_layout
        if not isinstance(x_type.layout, TiledTensorLayout):
            raise TypeError(
                "Invalid layout for partitioning register tensor. "
                f"got(x({x_type.layout}),expected(TiledTensorLayout))"
            )
        x_layout = x_type.layout
        x_shape = x_layout.shape()
        x_thrval_layout = canonicalize_thread_value_layout(x_layout.thrval_layout())
        x_thr_layout, x_val_layout = x_thrval_layout
        if coalesce(x_thr_layout) != coalesce(thr_layout):
            raise TypeError(
                "Invalid thread layout for partitioning register tensor. "
                f"got(x({x_thr_layout}),expected({thr_layout}))"
            )
        part, remain = group(x_val_layout, val_layout.size())
        layouts = [part]
        layouts.extend([TensorLayout(1) for _ in range(len(x_shape) - 1)])
        if remain.size() > 1:
            layouts.append(remain)
        partition_val_layout = make_layout(*layouts)
        ret_thrval_layout = make_layout(thr_layout, partition_val_layout)
        rest_shape, _ = validate_shape(x_shape, shape)
        if len(rest_shape) > 0:
            x_shape = shape + rest_shape
        atom = ThrValAtom("thread_block", x_shape, ret_thrval_layout)
        tiled_layout = TiledTensorLayout(atom)
        return tiled_tensor(x_type.dtype, tiled_layout, x_type.scope)
    else:
        _, val_layout = canonicalize_thread_value_layout(thrval_layout)
        x_shape = product_each(x_type.layout.shape)
        layouts = [val_layout]
        layouts.extend([TensorLayout(1) for _ in range(len(shape) - 1)])
        rest_shape, rest_layout = validate_shape(x_shape, shape)
        if len(rest_shape) > 0:
            for i in rest_layout:
                layouts.append(i)
        layout = auto_layout if is_auto_layout(x_type.layout) else x_type.layout.compose(make_layout(*layouts))
        return tiled_tensor(x_type.dtype, layout, x_type.scope)


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
        if is_auto_layout(x_type.layout):
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
        if is_auto_layout(x_type.layout):
            return tiled_tensor(x_type.dtype, auto_layout, x_type.scope)
        if is_auto_mma(self.tiled_mma):
            return tiled_tensor(x_type.dtype, auto_layout, x_type.scope)
        shape, thrval_layout = self.tiled_mma.b_tv_layout()
        return infer_type(x_type, shape, thrval_layout)


def partition_B(x: Expr, tiled_mma: TiledMma):
    return PartitionB(x, tiled_mma).make_call()
