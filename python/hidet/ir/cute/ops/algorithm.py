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
from typing import Callable, List, Optional, Dict

from hidet.ir.expr import Expr
from hidet.ir.type import BaseType, DataType

from hidet.ir.cute.expr import Op
from hidet.ir.cute.type import TiledTensorType, tiled_tensor, logical_encoding
from hidet.ir.cute.layout import TiledTensorLayout, is_auto_layout, make_layout, filter_lo_hi, compact_col_major
from hidet.ir.cute.expr import CConst


class InclusiveScan(Op):
    """
    Perform an inclusive scan over the given axis.

    Args:
        x: The input tensor to scan.
        init: The initial value for the scan.
        axis: The axis to scan over.
        scan_op: The scan operation to apply.
        tiled_layout: The layout of the input tensor. If not provided, the layout will be inferred in layout
        synthesis algorithm.
        update_init: Whether to update the initial value.
        scan_length: The length of the scan. If not provided, this operator will scan over all the elements
        in the given axis. If provided, this operator will only scan up to the given length.
    """

    def __init__(
        self,
        x: Expr,
        init: Expr,
        axis: int,
        scan_op: Callable[[Expr, Expr], Expr],
        tiled_layout: Optional[TiledTensorLayout] = None,
        update_init: Optional[bool] = False,
        scan_length: Optional[Expr] = None,
    ):
        super().__init__(
            args=[x, init],
            attrs={"axis": axis, "scan_op": scan_op, "tiled_layout": tiled_layout, "update_init": update_init},
        )
        self.x: Expr = x
        self.init: Expr = init
        self.axis: int = axis
        self.scan_op: Callable[[Expr, Expr], Expr] = scan_op
        self.tiled_layout: Optional[TiledTensorLayout] = tiled_layout
        self.update_init: bool = update_init
        self.scan_length: Optional[Expr] = scan_length

    def resolve_logical_encoding(self):
        assert self.tiled_layout is not None and isinstance(self.tiled_layout, TiledTensorLayout)
        shape = self.tiled_layout.shape()
        thr_layout = self.tiled_layout.thr_layout()
        val_layout = self.tiled_layout.val_layout()
        tv = make_layout(thr_layout, val_layout)
        enc = logical_encoding(shape, tv)
        cont_stride = compact_col_major(shape)
        lo = cont_stride[self.axis]
        hi = cont_stride[self.axis] * shape[self.axis]
        init_thr_layout = filter_lo_hi(thr_layout, lo, hi)
        init_val_layout = filter_lo_hi(val_layout, lo, hi)
        init_tv = make_layout(init_thr_layout, init_val_layout)
        init_enc = logical_encoding(shape, init_tv)
        return [enc, init_enc, enc]

    def infer_type(self, arg_types: List[BaseType]) -> BaseType:
        assert len(arg_types) == 2
        x_ty = arg_types[0]
        init_ty = arg_types[1]
        if not isinstance(x_ty, TiledTensorType):
            raise TypeError(f"input tensor must be a tiled tensor, but got {x_ty}")
        if not x_ty.scope.is_register():
            raise TypeError(f"input tensor must be a register tensor, but got {x_ty}")
        if not is_auto_layout(x_ty.layout) and not isinstance(x_ty.layout, TiledTensorLayout):
            raise TypeError(f"input tensor layout must be auto layout or tiled layout, but got {x_ty.layout}")
        if not isinstance(init_ty, TiledTensorType) and not isinstance(init_ty, DataType):
            raise TypeError(f"init tensor must be a tiled tensor or a data type, but got {init_ty}")
        if isinstance(init_ty, TiledTensorType):
            if not init_ty.scope.is_register():
                raise TypeError(f"init tensor must be a register tensor, but got {init_ty}")
            if not is_auto_layout(init_ty.layout) and not isinstance(init_ty.layout, TiledTensorLayout):
                raise TypeError(f"init tensor layout must be auto layout or tiled layout, but got {init_ty.layout}")
            init_dtype = init_ty.dtype
        else:
            init_dtype = init_ty
        if not x_ty.dtype == init_dtype:
            raise TypeError(
                f"input tensor and init tensor must have the same data type, but got {x_ty.dtype} and {init_dtype}"
            )
        if not is_auto_layout(x_ty.layout):
            shape = x_ty.layout.shape()
            axis = self.axis
            if axis < 0 or axis >= len(shape):
                raise ValueError(f"axis must be in the range of [0, {len(shape)})")
        return tiled_tensor(x_ty.dtype, x_ty.layout, x_ty.scope)

    def reforward(
        self, args: List[Expr], attrs_update: Dict[str, CConst] = None, annotations_update: Dict[str, CConst] = None
    ):
        attrs = self.attrs.copy()
        annotations = self.annotations.copy()
        if attrs_update is not None:
            attrs.update(attrs_update)
        if annotations_update is not None:
            annotations.update(annotations_update)
        ret = self.__class__(*args, **attrs, scan_length=self.scan_length)
        ret.annotations = annotations
        return ret


def inclusive_scan(
    x: Expr,
    axis: int,
    init: Expr,
    scan_op: Callable[[Expr, Expr], Expr],
    layout: Optional[TiledTensorLayout] = None,
    update_init: Optional[bool] = False,
    scan_length: Optional[Expr] = None,
):
    return InclusiveScan(x, init, axis, scan_op, layout, update_init, scan_length).make_call()
