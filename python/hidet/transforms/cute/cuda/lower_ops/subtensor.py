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
from typing import List, Union
from hidet.ir.expr import Expr
from hidet.ir.type import PointerType
from hidet.ir.dtypes import u64
from hidet.ir.tools import infer_type

from hidet.ir.cute.int_tuple import has_none, product_each
from hidet.ir.cute.ops.subtensor import SubTensor
from hidet.ir.cute import slice_and_offset, TensorLayout, TiledTensorLayout
from hidet.ir.cute.layout import register_tensor_layout

from .registry import OpEmitter, Buffer, register_impl


@register_impl(SubTensor)
class SubTensorEmitter(OpEmitter):
    def emit(self, op: SubTensor, args: List[Union[Buffer, Expr]], output: Buffer):
        assert isinstance(args[0], Buffer)
        src: Buffer = args[0]
        dst: Buffer = output
        src_buf = src.buffer
        src_off = src.offset
        src_ty = infer_type(src_buf)
        assert isinstance(src_ty, PointerType) or src_ty == u64
        coords = args[1]
        coords = coords[0] if len(coords) == 1 else coords
        layout = (
            register_tensor_layout(op.get_value_layout(src.layout))
            if isinstance(src.layout, TiledTensorLayout)
            else src.layout
        )
        if has_none(coords):
            _, offset = slice_and_offset(args[1], layout)
        else:
            offset = layout(coords)
        dst.buffer = self.auto_var(hint=op.name, e=src_buf)
        from hidet.ir.dtypes import i32

        if src_off is None and offset == 0:
            dst.offset = i32(0)
        else:
            dst.offset = self.auto_var(hint=op.name, e=src_off + offset if src_off is not None else i32(offset))

        if src.is_tma_buffer():
            assert src.scope.is_global()
            tile_shape = src.layout[0].shape_tuple
            tile_shape = product_each(tile_shape)
            assert len(tile_shape) == len(src.coords)
            rank = len(tile_shape)
            crd_layout = TensorLayout(src.layout[rank:].shape_tuple)
            crd = coords[rank:]
            dst.tensor_maps = src.tensor_maps
            dst.coords = src.coords[:-1] + [src.coords[-1] + crd_layout(crd) * tile_shape[-1]]
