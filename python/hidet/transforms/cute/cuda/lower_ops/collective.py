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
from typing import List, Union, Optional
from hidet.ir.expr import Expr, var
from hidet.ir.type import TensorPointerType, TensorType
from hidet.ir.tools import infer_type

from hidet.ir.cute.layout import TensorLayout
from hidet.ir.cute.int_tuple import compact_row_major
from hidet.ir.cute.collective import CollectiveStore
from hidet.ir.cute.ops import copy, partition_src, partition_dst, tiled_tensor_view, mask
from .registry import OpEmitter, Buffer, register_impl


@register_impl(CollectiveStore)
class CollectiveStoreEmitter(OpEmitter):
    def emit(self, op: CollectiveStore, args: List[Union[Buffer, Expr]], output: Optional[Buffer]):
        assert isinstance(args[1], Expr)
        src = args[0]
        dst: Expr = args[1]
        src: Expr = src.buffer if isinstance(src, Buffer) else src
        dim_extents = 2 + len(op.offsets)
        offsets = args[2:dim_extents]

        dst_ty = infer_type(dst)
        assert isinstance(dst_ty, (TensorType, TensorPointerType))
        if isinstance(dst_ty, TensorType):
            ttype = dst_ty
        else:
            ttype = dst_ty.tensor_type
        shape = ttype.shape if ttype.shape else ttype.layout.shape
        tile_shape, _ = op.tiled_copy.src_tv_layout()
        stride = compact_row_major(tuple(shape))
        tile_rank = len(tile_shape)
        mem_layout = TensorLayout(tile_shape, stride[-tile_rank:])
        t_mem = self.auto_var(hint="t_mem", e=tiled_tensor_view(~dst[offsets], mem_layout, "global"))
        src_expr, dst_expr = [partition_src(src, op.tiled_copy), partition_dst(t_mem, op.tiled_copy)]
        src_ty, dst_ty = [infer_type(src_expr), infer_type(dst_expr)]
        src_var, dst_var = [var("src", src_ty), var("dst", dst_ty)]
        self.declare(src_var, src_expr)
        self.declare(dst_var, dst_expr)
        if len(args) > dim_extents:
            extents = args[dim_extents:]
            masks_expr = mask(op.tiled_copy, extents)
            masks_ty = infer_type(masks_expr)
            masks_var = var("masks", masks_ty)
            self.declare(masks_var, masks_expr)
            self.append(copy(op.tiled_copy, src_var, dst_var, masks_var))
        else:
            self.append(copy(op.tiled_copy, src_var, dst_var))
