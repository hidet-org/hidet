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

from hidet.ir.cute.layout import TiledTensorLayout, TensorLayout
from hidet.ir.expr import Expr
from hidet.ir.cute.expr import Op
from hidet.ir.cute.type import tiled_tensor
from hidet.ir.type import BaseType, PointerType, TensorType, TensorPointerType

from hidet.ir.stmt import DeclareScope


class TiledTensorView(Op):
    def __init__(self, x: Expr, layout: Union[TiledTensorLayout, TensorLayout], scope: Union[DeclareScope, str] = None):
        if isinstance(scope, str):
            scope = DeclareScope.from_str(scope)
        super().__init__(args=[x], attrs={"layout": layout, "scope": scope})
        self.x: Expr = x
        assert (isinstance(layout, TiledTensorLayout) and scope == DeclareScope.Register) or isinstance(
            layout, TensorLayout
        )
        self.layout = layout
        self.scope = scope

    def infer_type(self, arg_types: List[BaseType]) -> BaseType:
        x_type = arg_types[0]
        assert isinstance(x_type, (TensorType, TensorPointerType, PointerType))
        import math

        if isinstance(x_type, TensorPointerType):
            ttype = x_type.tensor_type
            dtype = ttype.dtype
            tensor_size = None
        elif isinstance(x_type, TensorType):
            ttype = x_type
            dtype = ttype.dtype
            tensor_size = math.prod(ttype.shape)
        else:
            dtype = x_type.base_type
            tensor_size = None
        if isinstance(self.layout, TiledTensorLayout):
            val_layout = self.layout.val_layout()
            assert tensor_size is None or tensor_size == val_layout.size()
        else:
            assert isinstance(self.layout, TensorLayout)
            assert tensor_size is None or tensor_size == self.layout.size()
        return tiled_tensor(dtype=dtype, layout=self.layout, scope=self.scope)


def tiled_tensor_view(x: Expr, layout: Union[TiledTensorLayout, TensorLayout], scope: Union[DeclareScope, str] = None):
    return TiledTensorView(x, layout, scope).make_call()
