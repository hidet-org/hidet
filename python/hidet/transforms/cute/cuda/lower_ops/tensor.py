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
from hidet.ir.type import PointerType, TensorPointerType, TensorType
from hidet.ir.tools import infer_type

from hidet.ir.cute.ops.tensor import Tensor, TensorView
from hidet.ir.cute.layout import TiledTensorLayout, TensorLayout, ComposedTensorLayout, is_auto_layout, filter

from .registry import OpEmitter, Buffer, register_impl


@register_impl(Tensor)
class TensorEmitter(OpEmitter):
    def request_smem_nbytes(self, op: Tensor):
        if op.scope.is_shared():
            assert not is_auto_layout(op.layout)
            return filter(op.layout).size() * op.dtype.nbits // 8
        else:
            return 0

    def emit(self, op: Tensor, args: List[Union[Buffer, Expr]], output: Buffer):
        if op.scope.is_shared():
            output.buffer = self.auto_var(hint=op.name, e=self.get_smem_ptr(op, op.dtype, 0))
        elif op.scope.is_register():
            assert output.buffer is not None
        else:
            assert False, "unreachable"


@register_impl(TensorView)
class TensorViewEmitter(OpEmitter):
    def emit(self, op: TensorView, args: List[Union[Buffer, Expr]], output: Buffer):
        src: Union[Buffer, Expr] = args[0] if isinstance(args[0], Expr) else args[0].buffer
        src_ty = infer_type(src)
        assert isinstance(src_ty, (TensorType, TensorPointerType, PointerType))
        import math

        if isinstance(src_ty, TensorPointerType):
            ttype = src_ty.tensor_type
            tensor_size = None
            indices = [0] * len(ttype.shape)
        elif isinstance(src_ty, TensorType):
            ttype = src_ty
            tensor_size = math.prod(ttype.shape)
            indices = [0] * len(ttype.shape)
        else:
            tensor_size = None
        if isinstance(op.layout, TiledTensorLayout):
            assert tensor_size is None or tensor_size == op.layout.val_count()
            offset = 0
        else:
            assert isinstance(op.layout, (TensorLayout, ComposedTensorLayout))
            assert tensor_size is None or tensor_size == op.layout.size()
            if isinstance(op.layout, ComposedTensorLayout):
                apply_layout = op.layout.layout
            else:
                apply_layout = op.layout
            offset = apply_layout(tuple(op.tile_coords))
        if isinstance(src_ty, (TensorType, TensorPointerType)):
            output.buffer = self.auto_var(hint=op.name, e=~src[indices])
            output.offset = offset
        else:
            output.buffer = self.auto_var(hint=op.name, e=src)
            output.offset = offset

        if output.is_tma_buffer():
            assert op.tile_coords is not None and len(op.tile_coords) > 0
            output.coords = op.tile_coords
