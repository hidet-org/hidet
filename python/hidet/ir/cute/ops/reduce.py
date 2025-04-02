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
from typing import Dict, List, Tuple, Callable

from hidet.ir.type import BaseType
from hidet.ir.expr import Expr, Constant

from hidet.ir.cute import product
from hidet.ir.cute.layout import ThrValAtom, TensorLayout, make_layout, auto_layout, is_auto_layout, filter_lo_hi
from hidet.ir.cute.layout import AutoLayout
from hidet.ir.cute.type import tiled_tensor, TiledTensorType, TiledTensorLayout
from hidet.ir.cute.expr import Op, CConst


class Reduce(Op):
    """
    Base class for reduction operations.

    Attributes:
        x (Expr): The input expression.
        axis (int): The axis along which to reduce.
        op (Callable[[Expr, Expr], Expr]): The reduction operation to apply.
    """

    def __init__(self, x: Expr, axis: int, op: Callable[[Expr, Expr], Expr]):
        """
        Initialize the Reduce operation.

        Args:
            x (Expr): The input expression.
            axis (int): The axis along which to reduce.
            op (Callable[[Expr, Expr], Expr]): The reduction operation to apply.
        """
        super().__init__(args=[x], attrs={"axis": axis, "op": op})
        self.x: Expr = x
        self.axis: int = axis
        self.op: Callable[[Expr, Expr], Expr] = op

    def init(self) -> Constant:
        """
        Get the initial value for the reduction.
        For example, the initial value for a sum reduction is 0.

        Returns:
            Constant: The initial value for the reduction.
        """
        return NotImplemented()

    def resolve_logical_encoding(self):
        """
        Resolves the logical encoding for reduction operations in the Hexcute system.

        In the layout inference system, this function serves as a bridge to inject pre-resolved layout
        information into the layout synthesis process. When an operator's input and output layouts can
        be determined based on the operation's properties (before layout synthesis), this function
        provides that information to guide the layout inference.

        For reduction operator, if its input layout is already known, it can be considered resolved.
        The output layout can be inferred before the layout synthesis. This is different from the
        layout synthesis process which uses global information from the entire program to determine
        optimal layouts. In contrast, resolve_logical_encoding uses only local information available
        from the operation itself and its immediate inputs.

        The function performs the following steps:
        1. Infers the type of the input tensor
        2. Extracts the shape and layout information from the input tensor
        3. Creates input layout encoding based on the current tensor layout
        4. Creates output layout encoding based on the inferred reduction layout
        5. Returns both encodings as a list

        This pre-resolved layout information helps the layout synthesis system by:
        - Providing explicit layout constraints for the reduction operation
        - Ensuring consistent layout handling across the computation graph
        - Optimizing the layout inference process by avoiding redundant computations

        Returns:
            List[Expr]: A list containing two logical encodings:
                - First element: The logical encoding for the input tensor layout
                - Second element: The logical encoding for the output tensor layout after reduction

        Raises:
            AssertionError: If the input tensor has an auto layout or is not a TiledTensorLayout
        """
        from hidet.ir.tools import infer_type
        from hidet.ir.cute.type import logical_encoding

        x_ty = infer_type(self.x)
        assert not is_auto_layout(x_ty.layout)
        assert isinstance(x_ty.layout, TiledTensorLayout)
        shape = x_ty.layout.shape()
        thr, val = x_ty.layout.thr_layout(), x_ty.layout.val_layout()
        in_layout = make_layout(thr, val)
        out_layout = self.infer_layout(shape, thr, val)
        out_layout = make_layout(out_layout.thr_layout(), out_layout.val_layout())
        in_enc = logical_encoding(shape, in_layout)
        out_enc = logical_encoding(shape, out_layout)
        return [in_enc, out_enc]

    def infer_layout(self, shape: Tuple[int, ...], thrd: TensorLayout, val: TensorLayout) -> TiledTensorLayout:
        """
        Infer the layout for the reduction operation.

        Args:
            shape (Tuple[int]): The shape of the input tensor.
            thrd (TensorLayout): The thread layout.
            val (TensorLayout): The value layout.

        Returns:
            TiledTensorLayout: The inferred tiled tensor layout.
        """

        ax = self.axis
        if ax < 0 or ax >= len(shape):
            raise ValueError(f"invalid axis for reduce operator.(shape:{shape},axis:{ax})")
        lo = product(shape[:ax])
        hi = lo * shape[ax]
        thrd = filter_lo_hi(thrd, lo, hi)
        val = filter_lo_hi(val, lo, hi)
        layout = make_layout(thrd, val)
        atom = ThrValAtom("thread_block", shape, layout)
        levels = []
        tiled_layout = TiledTensorLayout(atom, levels)
        return tiled_layout

    def infer_type(self, arg_types: List[BaseType]) -> BaseType:
        """
        Infer the type for the reduction operation.

        Args:
            arg_types (List[BaseType]): The argument types.

        Returns:
            BaseType: The inferred type.
        """

        x_type = arg_types[0]
        assert isinstance(x_type, TiledTensorType)
        assert isinstance(x_type.layout, (AutoLayout, TiledTensorLayout))
        if is_auto_layout(x_type.layout):
            return tiled_tensor(x_type.dtype, auto_layout, x_type.scope)
        shape = x_type.layout.shape()
        thrd, val = x_type.layout.thr_layout(), x_type.layout.val_layout()
        tiled_layout = self.infer_layout(shape, thrd, val)
        return tiled_tensor(x_type.dtype, tiled_layout, x_type.scope)

    def reforward(
        self, args: List[Expr], attrs_update: Dict[str, CConst] = None, annotations_update: Dict[str, CConst] = None
    ):
        """
        Re-forward the reduction operation with updated arguments and attributes.

        Args:
            args (List[Expr]): The updated arguments.
            attrs_update (Dict[str, CConst], optional): Updated attributes. Defaults to None.
            annotations_update (Dict[str, CConst], optional): Updated annotations. Defaults to None.

        Returns:
            Reduce: The updated Reduce operation.
        """

        attrs = self.attrs.copy()
        annotations = self.annotations.copy()
        if attrs_update is not None:
            attrs.update(attrs_update)
        if annotations_update is not None:
            annotations.update(annotations_update)
        assert len(args) == 1
        if self.__class__ is not Reduce:
            attrs.pop("op", None)
        ret = self.__class__(*args, **attrs)
        ret.annotations = annotations
        return ret


class ReduceSum(Reduce):
    """
    Reduction operation that computes the sum of elements along a specified axis.
    """

    def __init__(self, x: Expr, axis: int):
        """
        Initialize the ReduceSum operation.

        Args:
            x (Expr): The input expression.
            axis (int): The axis along which to reduce.
        """

        def add(a: Expr, b: Expr):
            return a + b

        super().__init__(x, axis, add)

    def init(self):
        """
        Get the initial value for sum reduction.

        Returns:
            Constant: The initial value for the reduction (zero).
        """

        from hidet.ir.tools import infer_type

        x_type = infer_type(self.x)
        x_dtype = x_type.dtype
        return x_dtype.zero


class ReduceMean(Reduce):
    """
    Reduction operation that computes the mean of elements along a specified axis.
    """

    def __init__(self, x: Expr, axis: int):
        """
        Initialize the ReduceMean operation.

        Args:
            x (Expr): The input expression.
            axis (int): The axis along which to reduce.
        """

        def add(a: Expr, b: Expr):
            return a + b

        super().__init__(x, axis, add)

    def init(self) -> Constant:
        """
        Get the initial value for mean reduction.

        Returns:
            Constant: The initial value for the reduction (zero).
        """

        from hidet.ir.tools import infer_type

        x_type = infer_type(self.x)
        x_dtype = x_type.dtype
        return x_dtype.zero


class ReduceMax(Reduce):
    """
    Reduction operation that computes the maximum of elements along a specified axis.
    """

    def __init__(self, x: Expr, axis: int):
        """
        Initialize the ReduceMax operation.

        Args:
            x (Expr): The input expression.
            axis (int): The axis along which to reduce.
        """

        from hidet.ir.primitives import math

        def max(a: Expr, b: Expr):
            return math.max(a, b)

        super().__init__(x, axis, max)

    def init(self) -> Constant:
        """
        Get the initial value for max reduction.

        Returns:
            Constant: The initial value for the reduction (negative max value).
        """

        from hidet.ir.tools import infer_type

        x_type = infer_type(self.x)
        x_dtype = x_type.dtype
        return -x_dtype.max_value


class ReduceMin(Reduce):
    """
    Reduction operation that computes the minimum of elements along a specified axis.
    """

    def __init__(self, x: Expr, axis: int):
        """
        Initialize the ReduceMin operation.

        Args:
            x (Expr): The input expression.
            axis (int): The axis along which to reduce.
        """

        from hidet.ir.primitives import math

        def min(a: Expr, b: Expr):
            return math.min(a, b)

        super().__init__(x, axis, min)

    def init(self):
        """
        Get the initial value for min reduction.

        Returns:
            Constant: The initial value for the reduction (max value).
        """

        from hidet.ir.tools import infer_type

        x_type = infer_type(self.x)
        x_dtype = x_type.dtype
        return x_dtype.max_value


def reduce(x: Expr, axis: int, op: Callable[Expr, Expr]):
    return Reduce(x, axis, op).make_call()


def reduce_max(x: Expr, axis: int):
    return ReduceMax(x, axis).make_call()


def reduce_min(x: Expr, axis: int):
    return ReduceMin(x, axis).make_call()


def reduce_sum(x: Expr, axis: int):
    return ReduceSum(x, axis).make_call()


def reduce_mean(x: Expr, axis: int):
    return ReduceMean(x, axis).make_call()
