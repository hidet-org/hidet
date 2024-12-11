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
from hidet.ir.cute.type import tiled_tensor, TiledTensorType
from hidet.ir.cute.layout import ThrValAtom, remove_lo_hi, filter_lo_hi, common_reshape
from hidet.ir.cute import TensorLayout, TiledTensorLayout, auto_layout, is_auto_layout, flatten, make_layout, product


class Broadcast(Op):
    """
    Broadcast a tensor that resides in registers according to a given target tensor. The input and the target tensor
    could have different extents in the broadcasted dimension.

    EExample 1:
        a = make_tensor(f16, (BM, BN), register)
        c = broadcast(a, t)
        # where t is the target tensor, e.g., t.shape = (BM, BN1), and BN is not equal to BN1.

    Example 2:
        a = make_tensor(f16, (BM,), register)
        b = a.expand(dim=1)
        c = b.broadcast((BM, BN))
        # Combining the above operator sequence into a single operation.
    TODO: maybe we should support more flexible language constructs like triton.
    Currently, we combine the above operator sequence into a single operation

    The input tensor could be a tensor distributed across the threads in a thread block.
    In this case, the layout is a TV layout, where T is the thread dimension and V is the value dimension.
    The tensor is broadcasted across one axis of the logical domain.

    Case 1: Different Shapes
        If the input tensor's shape is not identical to the target tensor's shape, the broadcast is performed across
        the axis where the shapes differ.

        Example:
            Suppose we have an input tensor of shape (16, 8) with TV layout:
                a = ((4, 8), (2, 2)):((0, 1), (0, 8))
            and a target tensor of shape (16, 16) with TV layout:
                b = ((4, 8), (4, 2)):((64, 1), (16, 8))

        Steps:
            1. Contiguous strides of the input tensor: (16, 8):(1, 16).
            2. For broadcasting, strides should be: (16, 8):(1, 0).
            3. Strides between 16 (lower bound) and 128 (upper bound) should be zero.
            4. Verify that the input tensor's strides are not within the range of 16 and 128 to ensure compatibility.

    Case 2: Identical Shapes
        If the shapes of the input tensor and the target tensor are the same, no need to derive the broadcasted axis.
        Just verify if the input tensor can be broadcasted to the target tensor.

        Example:
            Suppose we have an input tensor of shape (16, 8) with TV layout:
                a = ((4, 8), (2, 2)):((0, 1), (0, 8))
            and a target tensor of shape (16, 8) with TV layout:
                b = ((4, 8), (2, 2)):((32, 1), (16, 8))

        Conditions:
            1. The shape with non-zero strides of the input tensor and target tensor should be the same.
            2. The non-zero strides of the input tensor should match the target tensor.

    Attributes:
        x (Expr): The tensor to be broadcasted.
        target (Expr): The target tensor to which the input tensor is broadcasted.
    """

    def __init__(self, x: Expr, target: Expr):
        """
        Initializes the Broadcast operation with the given tensor and target tensor.

        Args:
            x (Expr): The tensor to be broadcasted.
            target (Expr): The target full-rank tensor.
        """
        super().__init__(args=[x, target])
        self.x: Expr = x
        self.target: Expr = target

    def check_and_process_input(self, mode: TensorLayout, lo: int, hi: int):
        """
        Checks and processes the input tensor layout.

        Args:
            mode (TensorLayout): The tensor layout.
            lo (int): The lower bound for the broadcast.
            hi (int): The upper bound for the broadcast.

        Returns:
            TensorLayout: The processed tensor layout.

        Raises:
            ValueError: If the broadcast conditions are not met.
        """
        flat_shape = flatten(mode.shape_tuple)
        flat_stride = flatten(mode.stride_tuple)
        shrink = hi // lo
        result_shape = []
        result_stride = []
        for s, d in zip(flat_shape, flat_stride):
            if 0 < d < lo:
                if s * d > lo:
                    raise ValueError(f"fail to broadcast.(mode:{mode},lo:{lo},hi:{hi})")
                result_shape.append(s)
                result_stride.append(d)
            elif lo <= d < hi:
                raise ValueError(f"fail to broadcast.(mode:{mode},lo:{lo},hi:{hi})")
            elif d >= hi:
                result_shape.append(s)
                result_stride.append(d // shrink)
        return TensorLayout(tuple(result_shape), tuple(result_stride))

    def check_and_process_target(self, mode: TensorLayout, lo: int, hi: int):
        """
        Checks and processes the target tensor layout.

        Args:
            mode (TensorLayout): The tensor layout.
            lo (int): The lower bound for the broadcast.
            hi (int): The upper bound for the broadcast.

        Returns:
            TensorLayout: The processed tensor layout.

        Raises:
            ValueError: If the broadcast conditions are not met.
        """
        result = remove_lo_hi(mode, lo, hi)
        if any(d == 0 for d in result.stride_tuple):
            raise ValueError(f"fail to broadcast.(mode:{mode},lo:{lo},hi:{hi})")
        return result

    def infer_layout(
        self,
        x_shape: Tuple[int],
        x_t: TensorLayout,
        x_v: TensorLayout,
        trg_shape: Tuple[int],
        trg_t: TensorLayout,
        trg_v: TensorLayout,
    ):
        """
        Infers the layout of the broadcast operation.

        Args:
            x_shape (Tuple[int]): Shape of the input tensor.
            x_t (TensorLayout): Thread layout of the input tensor.
            x_v (TensorLayout): Value layout of the input tensor.
            trg_shape (Tuple[int]): Shape of the target tensor.
            trg_t (TensorLayout): Thread layout of the target tensor.
            trg_v (TensorLayout): Value layout of the target tensor.

        Returns:
            TensorLayout: The inferred layout.

        Raises:
            ValueError: If there is a mismatch in shapes or layouts.
        """
        if x_shape == trg_shape:
            x_t, trg_t = common_reshape(x_t, trg_t)
            x_v, trg_v = common_reshape(x_v, trg_v)
            if x_t.shape != trg_t.shape or x_v.shape != trg_v.shape:
                raise ValueError(f"broadcast fail.(x:{make_layout(x_t, x_v)},trg({make_layout(trg_t, trg_v)}))")
            return make_layout(x_t, x_v)
        else:
            lx = len(x_shape)
            lt = len(trg_shape)
            if lx != lt:
                raise ValueError(f"shape mismatch for broadcast.(x:{x_shape},trg:{trg_shape})")
            broadcast_dim = None
            for i, (sx, st) in enumerate(zip(x_shape, trg_shape)):
                if sx != st:
                    if broadcast_dim is None:
                        broadcast_dim = i
                    else:
                        raise ValueError(f"shape mismatch for broadcast.(x:{x_shape},trg:{trg_shape})")
            lo = product(x_shape[:broadcast_dim])
            hi = lo * x_shape[broadcast_dim]
            xt = self.check_and_process_input(x_t, lo, hi)
            xv = self.check_and_process_input(x_v, lo, hi)
            lo = product(trg_shape[:broadcast_dim])
            hi = lo * trg_shape[broadcast_dim]
            trgt = self.check_and_process_target(trg_t, lo, hi)
            trgv = self.check_and_process_target(trg_v, lo, hi)
            if xt != trgt or xv != trgv:
                raise ValueError(f"broadcast fail.(x:{make_layout(x_t, x_v)},trg({make_layout(trg_t, trg_v)}))")
            trg_t = filter_lo_hi(trg_t, lo, hi)
            trg_v = filter_lo_hi(trg_v, lo, hi)
            return make_layout(trg_t, trg_v)

    def infer_type(self, arg_types: List[BaseType]) -> BaseType:
        """
        Infers the type of the operation based on the types of its arguments.

        Args:
            arg_types (List[BaseType]): List of argument types.

        Returns:
            BaseType: The inferred type of the operation.

        Raises:
            AssertionError: If the argument types are not as expected.
        """
        x_type = arg_types[0]
        trg_type = arg_types[1]
        assert isinstance(x_type, TiledTensorType) and isinstance(trg_type, TiledTensorType)
        if is_auto_layout(x_type.layout) or is_auto_layout(trg_type.layout):
            return tiled_tensor(x_type.dtype, auto_layout, x_type.scope)
        assert isinstance(x_type.layout, TiledTensorLayout)
        assert isinstance(trg_type.layout, TiledTensorLayout)
        x_layout = x_type.layout
        trg_layout = trg_type.layout
        x_shape = x_layout.shape()
        trg_shape = trg_layout.shape()
        x_t, x_v = x_layout.thr_layout(), x_layout.val_layout()
        trg_t, trg_v = trg_layout.thr_layout(), trg_layout.val_layout()
        tv_lyt = self.infer_layout(x_shape, x_t, x_v, trg_shape, trg_t, trg_v)
        atom = ThrValAtom("thread_block", trg_shape, tv_lyt)
        tiled_layout = TiledTensorLayout(atom, [])
        return tiled_tensor(x_type.dtype, tiled_layout, x_type.scope)


def broadcast_to(x: Expr, target: Expr):
    return Broadcast(x, target).make_call()


class Transpose(Op):
    """
    Transpose a tensor that resides in the shared memory.

    This operator only transposes the view of the tensor, but doesn't reorder the data storage.

    The tensor can only be a tensor residing in shared memory.

    Attributes:
        x (Expr): The tensor to be transposed.
        dims (List[int]): The dimensions to transpose.
    """

    def __init__(self, x: Expr, dims: List[int]):
        """
        Initializes the Transpose operation with the given tensor and dimensions.

        Args:
            x (Expr): The tensor to be transposed.
            dims (List[int]): The dimensions to transpose.
        """
        super().__init__(args=[x], attrs={"dims": dims})
        self.x: Expr = x
        self.dims: List[int] = dims

    def infer_type(self, arg_types: List[BaseType]) -> BaseType:
        """
        Infers the type of the operation based on the types of its arguments.

        Args:
            arg_types (List[BaseType]): List of argument types.

        Returns:
            BaseType: The inferred type of the operation.

        Raises:
            TypeError: If the input type is not as expected.
            ValueError: If there is a mismatch in the dimensions.
        """
        x_type = arg_types[0]
        if not isinstance(x_type, TiledTensorType):
            raise TypeError("transpose requires the input to be a tensor")
        if not x_type.scope.is_shared():
            raise TypeError("transpose only supports tensors in shared memory")
        if is_auto_layout(x_type.layout):
            return tiled_tensor(x_type.dtype, auto_layout, x_type.scope)
        x_layout = x_type.layout
        shape_dims = len(x_layout.shape_tuple)
        num_dims = len(self.dims)
        if shape_dims != num_dims:
            raise ValueError(f"dimension mismatch.(got:{shape_dims},expected:{num_dims})")
        if any(dim >= num_dims for dim in self.dims):
            raise ValueError(f"invalid transpose dimensions.(got:{self.dims})")
        modes = []
        for i in self.dims:
            modes.append(x_layout[i])
        trans_layout = make_layout(*modes)
        return tiled_tensor(x_type.dtype, trans_layout, x_type.scope)


def transpose(x: Expr, *dims):
    return Transpose(x, list(dims)).make_call()
