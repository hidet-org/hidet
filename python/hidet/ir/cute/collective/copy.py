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

from hidet.ir.expr import Expr
from hidet.ir.cute.expr import Op
from hidet.ir.cute.type import TiledTensorType
from hidet.ir.cute.algorithm import TiledCopy
from hidet.ir.type import BaseType, void
from hidet.ir.cute.expr import CConst
from hidet.ir.type import TensorType, TensorPointerType
from hidet.ir.dtypes import i32
from hidet.ir.expr import is_constant


class CollectiveStore(Op):
    def __init__(self, tiled_copy: TiledCopy, src: Expr, dst: Expr, offsets: List[Expr], extents: List[Expr] = None):
        offsets = [i32(v) if is_constant(v) else v for v in offsets]
        super().__init__(
            args=[src, dst] + offsets + extents if extents else [src, dst] + offsets, attrs={"tiled_copy": tiled_copy}
        )
        self.tiled_copy: TiledCopy = tiled_copy
        self.src: Expr = src
        self.dst: Expr = dst
        self.offsets: List[Expr] = offsets
        self.extents: List[Expr] = extents

    def reforward(self, args: List[Expr], attrs_update: Dict[str, CConst] = None):
        attrs = self.attrs.copy()
        if attrs_update is not None:
            attrs.update(attrs_update)
        assert "tiled_copy" in attrs
        dim_extents = 2 + len(self.offsets)
        ret = self.__class__(attrs["tiled_copy"], args[0], args[1], args[2:dim_extents], args[dim_extents:])
        return ret

    def write_memory_op(self) -> bool:
        return True

    def infer_type(self, arg_types: List[BaseType]) -> BaseType:
        src_ty = arg_types[0]
        dst_ty = arg_types[1]
        if not (isinstance(src_ty, TiledTensorType) and isinstance(dst_ty, (TensorType, TensorPointerType))):
            raise TypeError(f"Type mismatch. (got:src({src_ty}),dst({dst_ty}))")
        return void


def collective_store(
    tiled_copy: TiledCopy, src: Expr, dst: Expr, offsets: List[Expr], extents: Optional[List[Expr]] = None
):
    return CollectiveStore(tiled_copy, src, dst, offsets, extents).make_call()
