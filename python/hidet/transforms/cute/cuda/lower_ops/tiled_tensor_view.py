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
from hidet.ir.expr import Expr, Var
from hidet.ir.type import PointerType, TensorPointerType, TensorType
from hidet.ir.tools import infer_type

from hidet.ir.cute.ops.tiled_tensor_view import TiledTensorView
from hidet.ir.cute.layout import TiledTensorLayout, TensorLayout, ComposedTensorLayout

from .registry import OpEmitter, Buffer, register_impl


@register_impl(TiledTensorView)
class TiledTensorViewEmitter(OpEmitter):
    def emit(self, op: TiledTensorView, args: List[Union[Buffer, Expr]], output: Buffer):
        src: Union[Buffer, Expr] = args[0] if isinstance(args[0], Expr) else args[0].buffer
        dst: Var = output.buffer
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
        else:
            assert isinstance(op.layout, (TensorLayout, ComposedTensorLayout))
            assert tensor_size is None or tensor_size == op.layout.size()
        if isinstance(src_ty, (TensorType, TensorPointerType)):
            self.assign(dst, ~src[indices])
        else:
            self.assign(dst, src)
        from hidet.ir.dtypes import i32

        if output.buffer is not None:
            output.offset = i32(0)
