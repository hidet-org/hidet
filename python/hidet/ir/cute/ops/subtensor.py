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

from typing import List, Tuple, Optional

from hidet.ir.type import BaseType
from hidet.ir.expr import Expr

from hidet.ir.cute.int_tuple import is_integer, flatten
from hidet.ir.cute.expr import Op
from hidet.ir.cute.type import tiled_tensor, TiledTensorType
from hidet.ir.cute import LayoutBase, TensorLayout, TiledTensorLayout, auto_layout, is_auto_layout
from hidet.ir.cute.layout import ThrValAtom, make_layout, coalesce


class SubTensor(Op):
    """
    The sub-tensor operation, which extracts a sub-tensor based on the given coordinates.
    The tensor could be a tensor residing in registers, shared memory, or global memory.
    For register tensors, the tensor could be distributed across the threads in a thread block.

    Example:
        a = make_tensor(f16, (BM, BK, stages), shared)
        b = a[:, :, 0]
    Note that we do not support the advanced slicing operation like a[2:4, :, 0], a[:4, :, 0] or a[2:, :, 0] yet.

    Attributes:
        x (Expr): The input tensor expression.
        coord (Tuple[Optional[Expr], ...]): The coordinates for slicing the tensor.
    """

    def __init__(self, x: Expr, coord: Tuple[Optional[Expr], ...]):
        """
        Initializes the SubTensor operation with the given tensor and coordinates.

        Args:
            x (Expr): The input tensor expression.
            coord (Tuple[Optional[Expr], ...]): The coordinates for slicing the tensor.
        """
        super().__init__(args=[x, coord])
        self.x: Expr = x
        self.coord: tuple = coord

    def get_value_layout(self, layout: TiledTensorLayout):
        """
        Get the value layout from the thread-value layout.
        """
        thrval_layout = layout.thrval_layout()
        val_layout = thrval_layout[0][1]
        rest = coalesce(thrval_layout[1])
        assert rest == TensorLayout(1)
        return val_layout

    def infer_type(self, arg_types: List[BaseType]) -> BaseType:
        """
        Infers the type of the sub-tensor based on the types of its arguments.

        Args:
            arg_types (List[BaseType]): List of argument types.

        Returns:
            BaseType: The inferred type of the sub-tensor.

        Raises:
            TypeError: If there is a type mismatch or if the tensor layout is not as expected.
            ValueError: If the coordinates are invalid for slicing.
        """
        x_type = arg_types[0]
        if is_auto_layout(x_type.layout):
            return tiled_tensor(x_type.dtype, auto_layout, x_type.scope)
        if not isinstance(x_type, TiledTensorType):
            raise TypeError(f"Type mismatch.(got:x({x_type}))")
        if not isinstance(x_type.layout, LayoutBase):
            raise TypeError(f"Can't slice a distributed tensor.(got:x({x_type}))")
        coord = self.coord[0] if len(self.coord) == 1 else self.coord
        if isinstance(x_type.layout, TiledTensorLayout):
            thr_layout = x_type.layout.thr_layout()
            val_layout = self.get_value_layout(x_type.layout)
            val_layout = val_layout(coord)
            if is_integer(val_layout):
                val_layout = TensorLayout(1)
            x_shape = x_type.layout.shape()
            sub_shape_layout = TensorLayout(x_shape)(coord)
            sub_shape = flatten(sub_shape_layout.shape_tuple)
            atom = ThrValAtom("thread_block", sub_shape, make_layout(thr_layout, val_layout))
            tiled_layout = TiledTensorLayout(atom)
            return tiled_tensor(x_type.dtype, tiled_layout, x_type.scope)
        else:
            layout = x_type.layout(coord)
            if is_integer(layout):
                return tiled_tensor(x_type.dtype, TensorLayout(1), x_type.scope)
            if not isinstance(layout, LayoutBase):
                raise ValueError(f"Invalid coord({self.coord}) for SubTensor")
            return tiled_tensor(x_type.dtype, layout, x_type.scope)


def sub_tensor(x: Expr, coord: Tuple[Optional[Expr], ...]):
    return SubTensor(x, coord).make_call()
