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
from typing import List, Optional

from hidet.ir.cute.layout import TensorLayout
from hidet.ir.expr import Expr
from hidet.ir.cute.expr import Op
from hidet.ir.cute.type import tiled_tensor, TiledTensorType
from hidet.ir.cute.algorithm import TiledCopy
from hidet.ir.type import BaseType, void
from hidet.ir.dtypes import u32


class Mask(Op):
    def __init__(self, tiled_copy: TiledCopy, extents: List[Expr]):
        super().__init__(args=extents, attrs={"tiled_copy": tiled_copy})
        self.extents: List[Expr] = extents
        assert len(extents) == len(tiled_copy.copy_atom.shape)
        self.tiled_copy = tiled_copy

    def infer_type(self, arg_types: List[BaseType]) -> BaseType:
        _, src_tv_layout = self.tiled_copy.src_tv_layout()
        nr_masks = src_tv_layout[1].size()
        nr_regs = (nr_masks + u32.nbytes * 8 - 1) // (u32.nbytes * 8)
        return tiled_tensor(dtype=u32, layout=TensorLayout(nr_regs), scope="register")


def mask(tiled_copy: TiledCopy, extents: List[Expr]):
    return Mask(tiled_copy, extents).make_call()


class Copy(Op):
    def __init__(self, tiled_copy: TiledCopy, src: Expr, dst: Expr, mask_: Optional[Expr] = None):
        super().__init__(args=[src, dst] + ([mask_] if mask_ is not None else []), attrs={"tiled_copy": tiled_copy})
        self.tiled_copy: TiledCopy = tiled_copy
        self.src: Expr = src
        self.dst: Expr = dst
        self.mask: Optional[Expr] = mask_

    def write_memory_op(self) -> bool:
        return True

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
            isinstance(src_ty.layout, TensorLayout)
            and isinstance(dst_ty.layout, TensorLayout)
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
