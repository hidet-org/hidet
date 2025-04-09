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
from typing import Tuple, List, Dict

from hidet.ir.cute.layout import (
    TiledTensorLayout,
    ComposedTensorLayout,
    TensorLayout,
    LayoutBase,
    make_layout,
    coalesce,
)
from hidet.ir.expr import Expr
from hidet.ir.cute.expr import Op
from hidet.ir.cute.type import TiledTensorType, logical_encoding
from hidet.ir.cute.algorithm import TiledMma, is_auto_mma
from hidet.ir.type import BaseType, void
from hidet.ir.cute.expr import CConst


class Mma(Op):
    """
    Class representing a Matrix Multiply-Accumulate (MMA) operation in a tiled layout.

    Attributes:
        d (Expr): The output tensor expression.
        a (Expr): The first input tensor expression.
        b (Expr): The second input tensor expression.
        c (Expr): The accumulation tensor expression.
        tiled_mma (TiledMma): The tiled MMA configuration.
    """

    def __init__(self, tiled_mma: TiledMma, d: Expr, a: Expr, b: Expr, c: Expr):
        """
        Initialize the Mma operation.

        Args:
            tiled_mma (TiledMma): The tiled MMA configuration.
            d (Expr): The output tensor expression.
            a (Expr): The first input tensor expression.
            b (Expr): The second input tensor expression.
            c (Expr): The accumulation tensor expression.
        """

        super().__init__(args=[d, a, b, c], attrs={"tiled_mma": tiled_mma})
        self.d: Expr = d
        self.a: Expr = a
        self.b: Expr = b
        self.c: Expr = c
        self.tiled_mma: TiledMma = tiled_mma

    def reforward(
        self, args: List[Expr], attrs_update: Dict[str, CConst] = None, annotations_update: Dict[str, CConst] = None
    ):
        """
        Re-forward the MMA operation with updated arguments and attributes.

        Args:
            args (List[Expr]): The updated arguments.
            attrs_update (Dict[str, CConst], optional): Updated attributes. Defaults to None.
            annotations_update (Dict[str, CConst], optional): Updated annotations. Defaults to None.

        Returns:
            Mma: The updated Mma operation.
        """

        attrs = self.attrs.copy()
        annotations = self.annotations.copy()
        if attrs_update is not None:
            attrs.update(attrs_update)
        if annotations_update is not None:
            annotations.update(annotations_update)
        assert "tiled_mma" in attrs
        ret = self.__class__(attrs["tiled_mma"], *args)
        ret.annotations = annotations
        return ret

    def resolve_logical_encoding(self):
        assert not is_auto_mma(self.tiled_mma)
        shape_mn, c_tv_layout = self.tiled_mma.c_tv_layout()
        shape_mk, a_tv_layout = self.tiled_mma.a_tv_layout()
        shape_nk, b_tv_layout = self.tiled_mma.b_tv_layout()
        _, d_tv_layout = self.tiled_mma.d_tv_layout()

        def to_logical_encoding(shape: Tuple[int], tv_layout: TensorLayout):
            t, v = tv_layout[0][0], coalesce(make_layout(tv_layout[0][1], tv_layout[1]))
            return logical_encoding(shape, make_layout(t, v))

        d_enc = to_logical_encoding(shape_mn, d_tv_layout)
        a_enc = to_logical_encoding(shape_mk, a_tv_layout)
        b_enc = to_logical_encoding(shape_nk, b_tv_layout)
        c_enc = to_logical_encoding(shape_mn, c_tv_layout)
        return [d_enc, a_enc, b_enc, c_enc]

    def infer_type(self, arg_types: List[BaseType]) -> BaseType:
        """
        Infer the type of the MMA operation based on the argument types.

        Args:
            arg_types (List[BaseType]): The argument types.

        Returns:
            BaseType: The inferred type (void in this case).
        """

        d_ty, a_ty, b_ty, c_ty = arg_types
        if any(not isinstance(ty, TiledTensorType) for ty in arg_types):
            raise TypeError(f"Type mimatch. (got:d({d_ty}),a({a_ty}),b({b_ty}),c({c_ty}))")
        if any(not isinstance(ty.layout, LayoutBase) for ty in arg_types):
            raise TypeError(
                f"Invalid layout. (got:d({d_ty.layout}),a({a_ty.layout}),b({b_ty.layout}),c({c_ty.layout}))"
            )
        _, c_tv_layout = self.tiled_mma.c_tv_layout()
        _, a_tv_layout = self.tiled_mma.a_tv_layout()
        _, b_tv_layout = self.tiled_mma.b_tv_layout()
        _, d_tv_layout = self.tiled_mma.d_tv_layout()

        from hidet.ir.cute import flatten

        def constraint(got, expected, operand: str):
            expt_tv = expected[0][0], make_layout(expected[0][1], expected[1])
            if isinstance(got, (TensorLayout, ComposedTensorLayout)):
                val = expt_tv[1]
                val_shape = flatten(val.shape_tuple)
                got_shape = flatten(got.shape_tuple)
                val_shape = tuple(filter(lambda x: x > 1, val_shape))
                got_shape = tuple(filter(lambda x: x > 1, got_shape))
                if got_shape != val_shape:
                    raise TypeError(f"Shape {operand} mismatch. (got:{got_shape},expected:{val_shape})")
            elif isinstance(got, TiledTensorLayout):
                got_tv = got.thr_layout(), got.val_layout()
                if any(coalesce(x) != coalesce(y) for x, y in zip(got_tv, expt_tv)):
                    raise TypeError(f"Layout {operand} mismatch. (got:{got.thrval_layout()},expected:{expected})")

        if a_ty.scope.is_register():
            constraint(a_ty.layout, a_tv_layout, "a")
        if b_ty.scope.is_register():
            constraint(b_ty.layout, b_tv_layout, "b")
        constraint(c_ty.layout, c_tv_layout, "c")
        constraint(d_ty.layout, d_tv_layout, "d")

        return void


def mma(tiled_mma: TiledMma, d: Expr, a: Expr, b: Expr, c: Expr):
    return Mma(tiled_mma, d, a, b, c).make_call()


class WgmmaFenceOperand(Op):
    def __init__(self, x: Expr):
        super().__init__(args=[x], attrs={})
        self.x: Expr = x

    def infer_type(self, arg_types: List[BaseType]) -> BaseType:
        if not isinstance(arg_types[0], TiledTensorType):
            raise TypeError(f"Type mismatch. (got:{arg_types[0]})")
        return void


def wgmma_fence_operand(x: Expr):
    return WgmmaFenceOperand(x).make_call()
