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
from typing import Union, List, Callable, Any, Tuple, Dict
from functools import partial

from hidet.ir.type import BaseType, DataType, TensorType, TensorPointerType, FuncType, void
from hidet.ir.expr import Expr, Var, Constant, PyScalar, var, convert
from hidet.ir.func import Function

from hidet.ir.cute.layout import (
    ThrValAtom,
    TiledTensorLayout,
    TensorLayout,
    make_layout,
    coalesce,
    composition,
    compact_col_major,
    common_reshape,
    auto_layout,
    is_auto_layout,
)
from hidet.ir.cute.type import tiled_tensor, TiledTensorType, LogicalEncoding, logical_encoding
from hidet.ir.cute.expr import Op, CConst


def local_broadcast(layouts: List[TensorLayout]):
    from hidet.ir.utils import broadcast_shapes
    from hidet.ir.cute import flatten

    shapes = [flatten(layout.shape_tuple) for layout in layouts]
    shape = tuple(broadcast_shapes(shapes))
    return TensorLayout(shape)


def distributed_broadcast(shapes: List[Tuple], layouts: List[TiledTensorLayout]):
    """
    Broadcasts shapes and layouts to ensure they are compatible for element-wise operations.

    Args:
        shapes (List[Tuple]): List of shapes of the input tensors.
        layouts (List[TiledTensorLayout]): List of layouts of the input tensors.

    Returns:
        Tuple: A tuple containing the broadcasted shape, thread layouts, and value layouts.
    """

    from hidet.ir.utils import broadcast_shapes

    broadcast_shape = tuple(broadcast_shapes(shapes))
    broadcast_layouts = []

    for shape in shapes:
        pos = len(broadcast_shape) - len(shape)
        shp = broadcast_shape[pos:]
        strd = compact_col_major(broadcast_shape)[pos:]
        broadcast_layouts.append(coalesce(TensorLayout(shp, strd)))

    thr_layouts = [coalesce(composition(x, y.thr_layout())) for x, y in zip(broadcast_layouts, layouts)]
    val_layouts = [coalesce(composition(x, y.val_layout())) for x, y in zip(broadcast_layouts, layouts)]

    def broadcast(layouts: List[TensorLayout]):
        if len(layouts) == 1:
            return layouts
        layout_0, layout_1 = common_reshape(layouts[0], layouts[1])
        result_layouts = [layout_0, layout_1]
        for layout in layouts[2:]:
            layout, _ = common_reshape(layout, result_layouts[0])
            for i, ly in enumerate(result_layouts[1:]):
                ly, _ = common_reshape(ly, layout)
                result_layouts[i + 1] = ly
            result_layouts.append(layout)
        return result_layouts

    thr_layouts = broadcast(thr_layouts)
    val_layouts = broadcast(val_layouts)

    return broadcast_shape, thr_layouts, val_layouts


def broadcast_layout(layouts: List[TensorLayout]):
    """
    Deduces the broadcasted layout from the given layouts.

    Args:
        layouts (List[TensorLayout]): List of tensor layouts to deduce the broadcast layout from.

    Returns:
        TensorLayout: The deduced broadcasted layout.
    """

    shp = None
    strd = None
    for layout in layouts:
        if shp is None:
            shp = layout.shape_tuple
            strd = list(layout.stride_tuple)
        else:
            curshp = layout.shape_tuple
            curstrd = layout.stride_tuple
            if curshp != shp:
                raise TypeError(f"Cannot broadcast shapes, (got:{curshp},expected:{shp})")
            for i, d in enumerate(curstrd):
                if d != 0:
                    if strd[i] != 0 and strd[i] != d:
                        raise TypeError(
                            f"Cannot broadcast shape due to logical domain mismatching, (got:{curstrd},expected:{strd})"
                        )
                    strd[i] = d
    return TensorLayout(shp, tuple(strd))


class Arithmetic(Op):
    """
    Performs element-wise arithmetic operations on two or more tensors.

    Attributes:
        inputs (List[Expr]): List of input expressions (tensors) for the arithmetic operation.
        op (Callable[[Any], Any]): A callable that defines the arithmetic operation to be performed.

    Methods:
        broadcast(shapes: List[Tuple], layouts: List[TiledTensorLayout]) -> Tuple:
            Broadcasts shapes and layouts to ensure they are compatible for element-wise operations.

        deduce_broadcast_layout(layouts: List[TensorLayout]) -> TensorLayout:
            Deduces the broadcasted layout from the given layouts.

        infer_type(arg_types: List[BaseType]) -> BaseType:
            Infers the type of the result of the arithmetic operation based on input types.

        reforward(args: List[Expr], attrs_update: Dict[str, CConst] = None,
                  annotations_update: Dict[str, CConst] = None) -> 'Arithmetic':
            Creates a new instance of the class with updated attributes and annotations.

    """

    def __init__(self, inputs: List[Expr], op: Callable[[Any], Any]):
        """
        Initializes the Arithmetic class with the given inputs and operation.

        Args:
            inputs (List[Expr]): List of input expressions (tensors).
            op (Callable[[Any], Any]): The arithmetic operation to be performed.
        """

        super().__init__(args=inputs, attrs={"op": op})
        self.inputs: List[Expr] = inputs
        self.op: Callable[[Any], Any] = op

    def resolve_logical_encoding(self):
        encs: List[LogicalEncoding] = []
        from hidet.ir.tools import infer_type

        arg_types = [infer_type(arg) for arg in self.inputs]
        output_layout = Arithmetic.infer_layout(*arg_types)
        if is_auto_layout(output_layout):
            raise RuntimeError(
                "Cannot resolve the logical encoding for tensors "
                f"because the output layout cannot be inferred.(got:{output_layout})"
            )
        layouts = [arg_ty.layout for arg_ty in arg_types] + [output_layout]
        for layout in layouts:
            if isinstance(layout, TiledTensorLayout):
                shape = layout.shape()
                thr_layout = layout.thr_layout()
                val_layout = layout.val_layout()
                encs.append(logical_encoding(shape, make_layout(thr_layout, val_layout)))
            else:
                encs.append(None)
        return tuple(encs)

    @staticmethod
    def infer_layout(*arg_types) -> Union[TensorLayout, TiledTensorLayout]:
        """
        If the layouts of all the argument tensors are tiled tensor layouts,
        the tensors are distributed across the threads within a block, which
        means the tensors should be broadcasted according the entire tensor
        shape held by the thread block. If the layouts of all the argument
        tensors are tensor layouts, the tensors are local tensor held by one
        single thread, which means the tensors should be broadcasted locally.
        We assume the broadcast rules follow the convension of numpy
        multi-dimensional arrays.
        """
        if any(is_auto_layout(arg_ty.layout) for arg_ty in arg_types):
            return auto_layout

        distributed_arithmetic = None
        for arg_ty in arg_types:
            if not isinstance(arg_ty, TiledTensorType):
                raise TypeError(f"Type mismatch.(got:{arg_ty},expected:TiledTensorType)")
            elif not isinstance(arg_ty.layout, (TensorLayout, TiledTensorLayout)):
                raise TypeError(f"Type mismatch.(got:{arg_ty.layout},expected:TiledTensorLayout or TensorLayout)")
            elif not arg_ty.scope.is_register():
                raise TypeError(
                    f"Type mismatch(got:{arg_ty.scope},expected:Register) arithmetic operations only "
                    "support data stored in register files."
                )
            if distributed_arithmetic is None:
                distributed_arithmetic = isinstance(arg_ty.layout, TiledTensorLayout)
            else:
                checked = distributed_arithmetic and isinstance(arg_ty.layout, TiledTensorLayout)
                checked |= (not distributed_arithmetic) and isinstance(arg_ty.layout, TensorLayout)
                if not checked:
                    raise TypeError(
                        f"Type mismatch.(got:{arg_ty.layout},expected:"
                        "{TiledTensorLayout if distributed_arithmetic else TensorLayout})"
                    )

        if distributed_arithmetic:
            layouts = [arg.layout for arg in arg_types]
            shapes = [layout.shape() for layout in layouts]
            shape, thr_layouts, val_layouts = distributed_broadcast(shapes, layouts)
            thr_layout = broadcast_layout(thr_layouts)
            val_layout = broadcast_layout(val_layouts)
            tv_atom = ThrValAtom("thread_block", shape, make_layout(thr_layout, val_layout))
            return TiledTensorLayout(tv_atom)
        else:
            layouts = [arg.layout for arg in arg_types]
            return local_broadcast(layouts)

    def infer_type(self, arg_types: List[BaseType]) -> BaseType:
        """
        Infers the type of the result of the arithmetic operation based on input types.

        Args:
            arg_types (List[BaseType]): List of input types.

        Returns:
            BaseType: The inferred type of the result.
        """

        from hidet.ir.tools import infer_type

        if isinstance(self.op, Var):
            func = self.op
            if not isinstance(func.type, FuncType):
                raise TypeError(f"invalid elementwise function. (got:{func.type},expected:{Function})")
            func_type = func.type
            output_ty = func_type.param_types[-1]
            if isinstance(output_ty, TensorType):
                output_ty = output_ty.dtype
            elif isinstance(output_ty, TensorPointerType):
                output_ty = output_ty.tensor_type.dtype
        else:
            inputs = [var("v", arg_ty.dtype) for arg_ty in arg_types]
            output = self.op(*inputs)
            output_ty = infer_type(output)
            assert isinstance(output_ty, DataType)

        layout = Arithmetic.infer_layout(*arg_types)

        return tiled_tensor(output_ty, layout, "register")

    def reforward(
        self, args: List[Expr], attrs_update: Dict[str, CConst] = None, annotations_update: Dict[str, CConst] = None
    ):
        """
        Creates a new instance of the class with updated attributes and annotations.

        Args:
            args (List[Expr]): List of input expressions (tensors).
            attrs_update (Dict[str, CConst], optional): Dictionary of attributes to update. Defaults to None.
            annotations_update (Dict[str, CConst], optional): Dictionary of annotations to update. Defaults to None.

        Returns:
            Arithmetic: A new instance of the Arithmetic class with updated attributes and annotations.
        """

        attrs = self.attrs.copy()
        annotations = self.annotations.copy()
        if attrs_update is not None:
            attrs.update(attrs_update)
        if annotations_update is not None:
            annotations.update(annotations_update)
        ret = self.__class__(args, **attrs)
        ret.annotations = annotations
        return ret


class Cast(Arithmetic):
    def __init__(self, x, dtype: DataType):
        ir_cast = partial(Cast.ir_cast, dtype)

        Cast.ir_cast.__name__ = "cast"
        super().__init__([x], ir_cast)
        self.attrs.update({"dtype": dtype})

    @staticmethod
    def ir_cast(dtype, x):
        from hidet.ir import expr

        return expr.cast(x, dtype)

    def reforward(
        self, args: List[Expr], attrs_update: Dict[str, CConst] = None, annotations_update: Dict[str, CConst] = None
    ):
        attrs = self.attrs.copy()
        annotations = self.annotations.copy()
        if attrs_update is not None:
            attrs.update(attrs_update)
        if annotations_update is not None:
            annotations.update(annotations_update)
        assert "dtype" in attrs
        assert len(args) == 1
        x = args[0]
        dtype = attrs["dtype"]
        ret = self.__class__(x, dtype)
        ret.annotations = annotations
        return ret


class UnaryOp(Arithmetic):
    def __init__(self, x: Expr):
        super().__init__([x], self.scalar_op())

    def scalar_op(self):
        import hidet.ir.expr

        cls_map = {"Neg": hidet.ir.expr.Neg}

        cls_name = self.__class__.__name__

        if cls_name not in cls_map:
            raise NotImplementedError(f"No implementation for {cls_name} unary op")

        expr_cls = cls_map[cls_name]

        def unary(x):
            return Expr._unary(expr_cls, x)  # pylint: disable=protected-access

        unary.__name__ = unary.__name__ + f"_{cls_name}"
        return unary

    def reforward(
        self, args: List[Expr], attrs_update: Dict[str, CConst] = None, annotations_update: Dict[str, CConst] = None
    ):
        attrs = self.attrs.copy()
        annotations = self.annotations.copy()
        if attrs_update is not None:
            attrs.update(attrs_update)
        if annotations_update is not None:
            annotations.update(annotations_update)
        assert len(args) == 1
        ret = self.__class__(*args)
        ret.annotations = annotations
        return ret


class BinaryOp(Arithmetic):
    def __init__(self, x: Expr, y: Expr):
        super().__init__([x, y], self.scalar_op())

    def scalar_op(self):
        import hidet.ir.expr

        cls_name = self.__class__.__name__

        if not hasattr(hidet.ir.expr, cls_name):
            raise NotImplementedError(f"No implementation for {cls_name} binary op")
        expr_cls = getattr(hidet.ir.expr, cls_name)

        def binary(x, y):
            return Expr._binary(expr_cls, x, y)  # pylint: disable=protected-access

        binary.__name__ = binary.__name__ + f"_{cls_name}"
        return binary

    def reforward(
        self, args: List[Expr], attrs_update: Dict[str, CConst] = None, annotations_update: Dict[str, CConst] = None
    ):
        attrs = self.attrs.copy()
        annotations = self.annotations.copy()
        if attrs_update is not None:
            attrs.update(attrs_update)
        if annotations_update is not None:
            annotations.update(annotations_update)
        assert len(args) == 2
        ret = self.__class__(*args)
        ret.annotations = annotations
        return ret


class TernaryOp(Arithmetic):
    def __init__(self, x: Expr, y: Expr, z: Expr):
        super().__init__([x, y, z], self.scalar_op())

    def scalar_op(self):
        cls_name = self.__class__.__name__

        supported_ops = ("MultiplyAdd",)

        if cls_name not in supported_ops:
            raise NotImplementedError(f"No implementation for {cls_name} ternary op")

        def ternary(x: Expr, y: Expr, z: Expr):
            return x * y + z

        ternary.__name__ = ternary.__name__ + "_MutiplyAdd"
        return ternary

    def reforward(
        self, args: List[Expr], attrs_update: Dict[str, CConst] = None, annotations_update: Dict[str, CConst] = None
    ):
        attrs = self.attrs.copy()
        annotations = self.annotations.copy()
        if attrs_update is not None:
            attrs.update(attrs_update)
        if annotations_update is not None:
            annotations.update(annotations_update)
        assert len(args) == 3
        ret = self.__class__(*args)
        ret.annotations = annotations
        return ret


class Neg(UnaryOp):
    pass


class Add(BinaryOp):
    pass


class Sub(BinaryOp):
    pass


class Multiply(BinaryOp):
    pass


class Div(BinaryOp):
    pass


class MultiplyAdd(TernaryOp):
    pass


class Exp(UnaryOp):
    def scalar_op(self):
        from hidet.ir.primitives import math

        def ir_exp(x):
            return math.exp(x)

        return ir_exp


class Relu(UnaryOp):
    def scalar_op(self):
        from hidet.ir.primitives import math

        def ir_relu(x):
            from hidet.ir.tools import infer_type

            dtype = infer_type(x)
            assert isinstance(dtype, DataType)
            return math.max(x, dtype.zero)

        return ir_relu


class Silu(UnaryOp):
    def scalar_op(self):
        from hidet.ir.primitives import math

        def ir_silu(x):
            from hidet.ir.tools import infer_type

            dtype = infer_type(x)
            assert isinstance(dtype, DataType)
            return x / (dtype.one + math.exp(-x))

        return ir_silu


class RSqrt(UnaryOp):
    def scalar_op(self):
        from hidet.ir.primitives import math

        def ir_rsqrt(x):
            from hidet.ir.tools import infer_type

            dtype = infer_type(x)
            assert isinstance(dtype, DataType)
            return math.rsqrt(x)

        return ir_rsqrt


class ElementwiseMin(BinaryOp):
    def scalar_op(self):
        from hidet.ir.primitives import math

        def min(x, y):
            return math.min(x, y)

        return min


class ElementwiseMax(BinaryOp):
    def scalar_op(self):
        from hidet.ir.primitives import math

        def max(x, y):
            return math.max(x, y)

        return max


def arithmetic(*inputs, op: Callable[[Any], Any]):
    return Arithmetic(inputs, op).make_call()


def cast(x: Expr, dtype: DataType):
    return Cast(x, dtype).make_call()


def relu(x: Expr):
    return Relu(x).make_call()


def exp(x: Expr):
    return Exp(x).make_call()


def silu(x: Expr):
    return Silu(x).make_call()


def rsqrt(x: Expr):
    return RSqrt(x).make_call()


def elementwise_min(x: Expr, y: Expr):
    return ElementwiseMin(x, y).make_call()


def elementwise_max(x: Expr, y: Expr):
    return ElementwiseMax(x, y).make_call()


class Fill(Op):
    def __init__(self, x: Expr, val: Constant):
        super().__init__(args=[x], attrs={"val": val})
        self.x: Expr = x
        self.val: Constant = val

    def infer_type(self, arg_types: List[BaseType]) -> BaseType:
        x_type = arg_types[0]

        if not isinstance(x_type, TiledTensorType):
            raise TypeError(f"invalid input type.(got:{x_type},expected:{TiledTensorType})")
        if not x_type.scope.is_register():
            raise TypeError(f"Fill op only supports tensor in the register file.(got:{x_type})")
        return void


def fill(x: Expr, val: Union[PyScalar, Constant]):
    val = convert(val)
    return Fill(x, val).make_call()
