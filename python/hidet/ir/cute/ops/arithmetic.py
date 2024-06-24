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
from typing import List, Callable, Any, Tuple, Dict

from hidet.ir.type import BaseType, DataType
from hidet.ir.expr import Expr, var

from hidet.ir.cute.layout import (
    ThrValAtom,
    TiledTensorLayout,
    TensorLayout,
    make_layout,
    coalesce,
    composition,
    compact_col_major,
    common_reshape,
)
from hidet.ir.cute.type import tiled_tensor, TiledTensorType
from hidet.ir.cute.expr import Op, CConst


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

    def broadcast(self, shapes: List[Tuple], layouts: List[TiledTensorLayout]):
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

    def deduce_broadcast_layout(self, layouts: List[TensorLayout]):
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
                                "Cannot broadcast shape due to logical domain mismatching, "
                                f"(got:{curstrd},expected:{strd})"
                            )
                        strd[i] = d
        return TensorLayout(shp, tuple(strd))

    def infer_type(self, arg_types: List[BaseType]) -> BaseType:
        """
        Infers the type of the result of the arithmetic operation based on input types.

        Args:
            arg_types (List[BaseType]): List of input types.

        Returns:
            BaseType: The inferred type of the result.
        """

        for arg_ty in arg_types:
            if not isinstance(arg_ty, TiledTensorType):
                raise TypeError(f"Type mismatch.(got:{arg_ty},expected:TiledTensorType)")
            elif not isinstance(arg_ty.layout, TiledTensorLayout):
                raise TypeError(f"Type mismatch.(got:{arg_ty.layout},expected:TiledTensorLayout)")
            elif not arg_ty.scope.is_register():
                raise TypeError(
                    f"Type mismatch(got:{arg_ty.scope},expected:Register) "
                    "arithmetic operations only support data stored in register files."
                )

        layouts = [arg.layout for arg in arg_types]
        shapes = [layout.shape() for layout in layouts]
        shape, thr_layouts, val_layouts = self.broadcast(shapes, layouts)
        thr_layout = self.deduce_broadcast_layout(thr_layouts)
        val_layout = self.deduce_broadcast_layout(val_layouts)
        tv_atom = ThrValAtom("thread_block", shape, make_layout(thr_layout, val_layout))
        layout = TiledTensorLayout(tv_atom)

        from hidet.ir.tools import infer_type

        inputs = [var("v", arg_ty.dtype) for arg_ty in arg_types]
        output = self.op(*inputs)
        output_ty = infer_type(output)
        assert isinstance(output_ty, DataType)

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


def arithmetic(*inputs, op: Callable[[Any], Any]):
    return Arithmetic(inputs, op).make_call()
