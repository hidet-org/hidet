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

from hidet.ir.type import BaseType
from hidet.ir.expr import Expr
from hidet.ir.stmt import DeclareScope

from hidet.ir.cute.expr import Op
from hidet.ir.cute.type import tiled_tensor, TiledTensorType
from hidet.ir.cute import TensorLayout
from hidet.ir.cute.layout import TiledTensorLayout


class Rearrange(Op):
    def __init__(self, x: Expr, layout: Union[TiledTensorLayout, TensorLayout], scope: Union[DeclareScope, str] = None):
        if isinstance(scope, str):
            scope = DeclareScope.from_str(scope)
        elif scope is None:
            scope = DeclareScope.Default
        super().__init__(args=[x], attrs={"layout": layout, "scope": scope})
        self.x: Expr = x
        self.layout = layout
        self.scope: DeclareScope = scope

    def infer_type(self, arg_types: List[BaseType]) -> BaseType:
        x_type = arg_types[0]
        assert isinstance(x_type, TiledTensorType)
        return tiled_tensor(x_type.dtype, self.layout, self.scope)
