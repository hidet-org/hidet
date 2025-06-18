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
from hidet.ir.expr import Expr, cast, deref
from hidet.ir.dtypes.vector import VectorType
from hidet.ir.primitives import math
from hidet.ir.cute.ops.misc import Broadcast, Transpose, Pack, GetItem
from hidet.ir.cute.layout import TiledTensorLayout, register_tensor_layout

from .registry import OpEmitter, Buffer, register_impl


@register_impl(Broadcast)
class BroadcastEmitter(OpEmitter):
    def emit(self, op: Broadcast, args: List[Union[Buffer, Expr]], output: Buffer):
        assert isinstance(args[0], Buffer)
        src: Buffer = args[0]
        dst: Buffer = output
        assert src.scope.is_register()
        dst.buffer = src.buffer
        assert src.offset is None and dst.offset is None


@register_impl(Transpose)
class TransposeEmitter(OpEmitter):
    def emit(self, op: Transpose, args: List[Union[Buffer, Expr]], output: Buffer):
        assert isinstance(args[0], Buffer)
        src: Buffer = args[0]
        dst: Buffer = output
        assert src.scope.is_shared()
        dst.buffer = src.buffer
        assert src.offset is None and dst.offset is None


@register_impl(Pack)
class PackEmitter(OpEmitter):
    def emit(self, op: Pack, args: List[Union[Buffer, Expr]], output: Buffer):
        assert isinstance(args[0], Buffer)
        dst: Buffer = output
        assert dst.scope.is_register()
        assert isinstance(dst.layout, TiledTensorLayout)
        val_layout = dst.layout.val_layout()
        register_layout = register_tensor_layout(val_layout)
        extents = register_layout.shape
        from hidet.ir.type import TensorType, TensorPointerType, PointerType
        from hidet.ir.tools import infer_type

        def get_value(buffer: Expr, offset: Expr):
            buffer_ty = infer_type(buffer)
            if isinstance(buffer_ty, (TensorType, TensorPointerType)):
                return buffer[offset]
            else:
                assert isinstance(buffer_ty, PointerType)
                return deref(buffer + offset)

        with self.for_grid(extents) as indices:
            self.buffer_store(
                dst.buffer,
                [register_layout(indices)],
                math.make_vector(*[get_value(arg.buffer, register_layout(indices)) for arg in args]),
            )


@register_impl(GetItem)
class GetItemEmitter(OpEmitter):
    def emit(self, op: GetItem, args: List[Union[Buffer, Expr]], output: Buffer):
        assert isinstance(args[0], Buffer)
        src: Buffer = args[0]
        dst: Buffer = output
        vector_type = src.dtype
        assert isinstance(vector_type, VectorType)
        lane_type = vector_type.lane_type
        assert dst.scope.is_register()
        assert isinstance(dst.layout, TiledTensorLayout)
        val_layout = dst.layout.val_layout()
        register_layout = register_tensor_layout(val_layout)
        extents = register_layout.shape
        with self.for_grid(extents) as indices:
            val = src.buffer[register_layout(indices)]
            vector = cast(~val, ~lane_type)
            self.buffer_store(dst.buffer, [register_layout(indices)], deref(vector + op.index))
