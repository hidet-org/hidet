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
from typing import List, Dict, Optional

from hidet.ir.cute.layout import ComposedTensorLayout, TensorLayout
from hidet.ir.expr import Expr
from hidet.ir.cute.expr import Op
from hidet.ir.cute.type import tiled_tensor, TiledTensorType
from hidet.ir.cute.algorithm import TiledCopy
from hidet.ir.type import BaseType, void
from hidet.ir.dtypes import u32
from hidet.ir.cute.expr import CConst


class Mask(Op):
    """
    Create a mask tensor for copies.

    Attributes:
        extents (List[Expr]): extents of the dimensions of the tensor to be copied
        tiled_copy (TiledCopy): The tiled copy that represents the task mapping of the copy operation.

    Methods:
        reforward(args: List[Expr], attrs_update: Dict[str, CConst] = None,
                  annotations_update: Dict[str, CConst] = None) -> 'Mask':
            Creates a new instance of the Mask class with updated attributes and annotations.

        infer_type(arg_types: List[BaseType]) -> BaseType:
            Infers the type of the result of the mask operation based on input types.

    Example:
        Consider a copy operation that copies a tile of size (16, 16) from a source tensor to a destination tensor.
        The shape of the input tensor is (12, 12). Then, we should create a mask tensor for the copy operation to
        disable the elements that are out of bounds. The mask tensor can be created as follows:
        ```python
        mask = Mask([12, 12], tiled_copy)
        ```
    """

    def __init__(self, extents: List[Expr], tiled_copy: TiledCopy):
        """
        Initializes the Mask class with the given extents and tiled copy operation.

        Args:
            extents (List[Expr]): List of extents for the mask.
            tiled_copy (TiledCopy): The tiled copy operation associated with this mask.
        """

        super().__init__(args=extents, attrs={"tiled_copy": tiled_copy})
        self.extents: List[Expr] = extents
        assert len(extents) == len(tiled_copy.copy_atom.shape)
        self.tiled_copy = tiled_copy

    def reforward(
        self, args: List[Expr], attrs_update: Dict[str, CConst] = None, annotations_update: Dict[str, CConst] = None
    ):
        """
        Creates a new instance of the Mask class with updated attributes and annotations.

        Args:
            args (List[Expr]): List of input expressions (extents).
            attrs_update (Dict[str, CConst], optional): Dictionary of attributes to update. Defaults to None.
            annotations_update (Dict[str, CConst], optional): Dictionary of annotations to update. Defaults to None.

        Returns:
            Mask: A new instance of the Mask class with updated attributes and annotations.
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

    def infer_type(self, arg_types: List[BaseType]) -> BaseType:
        """
        Infers the type of the result of the mask operation based on input types.

        Args:
            arg_types (List[BaseType]): List of input types.

        Returns:
            BaseType: The inferred type of the result.
        """
        # FIXME:
        # Note that we can't infer the output type when the instruction selection pass hasn't been done.
        # So the "-1" here means the register count hasn't been determined, which can't be used to
        # generate code.
        annotations = self.annotations
        if len(annotations) == 0:
            return tiled_tensor(dtype=u32, layout=TensorLayout(-1), scope="register")
        rest_layout = annotations["rest_layout"]
        nr_masks = rest_layout.size()
        nr_regs = (nr_masks + u32.nbytes * 8 - 1) // (u32.nbytes * 8)
        return tiled_tensor(dtype=u32, layout=TensorLayout(nr_regs), scope="register")


def mask(tiled_copy: TiledCopy, extents: List[Expr]):
    return Mask(extents, tiled_copy).make_call()


class Copy(Op):
    """
    Copy operation that copies a tile from a source tensor to a destination tensor.

    Attributes:
        tiled_copy (TiledCopy): The tiled copy operation.
        src (Expr): The source tensor expression.
        dst (Expr): The destination tensor expression.
        mask (Optional[Expr]): Optional mask expression for the copy.

    Methods:
        reforward(args: List[Expr], attrs_update: Dict[str, CConst] = None,
            annotations_update: Dict[str, CConst] = None) -> 'Copy':
            Creates a new instance of the Copy class with updated attributes and annotations.

        write_memory_op() -> bool:
            Checks if the copy operation involves writing to memory.

        infer_type(arg_types: List[BaseType]) -> BaseType:
            Infers the type of the result of the copy operation based on input types.
    """

    def __init__(self, tiled_copy: TiledCopy, src: Expr, dst: Expr, mask_: Optional[Expr] = None):
        super().__init__(args=[src, dst] + ([mask_] if mask_ is not None else []), attrs={"tiled_copy": tiled_copy})
        self.tiled_copy: TiledCopy = tiled_copy
        self.src: Expr = src
        self.dst: Expr = dst
        self.mask: Optional[Expr] = mask_

    def reforward(
        self, args: List[Expr], attrs_update: Dict[str, CConst] = None, annotations_update: Dict[str, CConst] = None
    ):
        attrs = self.attrs.copy()
        annotations = self.annotations.copy()
        if attrs_update is not None:
            attrs.update(attrs_update)
        if annotations_update is not None:
            annotations.update(annotations_update)
        assert "tiled_copy" in attrs
        ret = self.__class__(attrs["tiled_copy"], *args)
        ret.annotations = annotations
        return ret

    def write_memory_op(self) -> bool:
        from hidet.ir.tools import infer_type

        dst_ty = infer_type(self.dst)
        return dst_ty.scope.is_memory()

    def infer_type(self, arg_types: List[BaseType]) -> BaseType:
        src_ty = arg_types[0]
        dst_ty = arg_types[1]
        mask_ty = arg_types[2] if len(arg_types) >= 3 else void
        if not (
            isinstance(src_ty, TiledTensorType)
            and isinstance(dst_ty, TiledTensorType)
            and (mask_ty is void or isinstance(mask_ty, TiledTensorType))
        ):
            raise TypeError(f"Type mismatch. (got:src({src_ty}),dst({dst_ty}),mask({mask_ty}))")
        if not (
            isinstance(src_ty.layout, (TensorLayout, ComposedTensorLayout))
            and isinstance(dst_ty.layout, (TensorLayout, ComposedTensorLayout))
            and (mask_ty is void or isinstance(mask_ty.layout, TensorLayout))
        ):
            raise TypeError(f"Invalid layout. (got:src({src_ty.layout}),dst({dst_ty.layout}),mask({mask_ty}))")
        src_size = src_ty.layout.size()
        dst_size = dst_ty.layout.size()
        if src_size != dst_size:
            raise TypeError(f"Tensor size mismatch. (got:src({src_size}),dst({dst_size}))")
        return void


def copy(tiled_copy: TiledCopy, src: Expr, dst: Expr, mask_: Optional[Expr] = None):
    return Copy(tiled_copy, src, dst, mask_).make_call()
