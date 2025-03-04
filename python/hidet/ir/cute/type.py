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
# pylint: disable=import-outside-toplevel
from typing import Union, Tuple
from hidet.ir.cute import TensorLayout, TiledTensorLayout, ComposedTensorLayout

from hidet.ir.type import BaseType, DataType
from hidet.ir.stmt import DeclareScope


class TiledTensorType(BaseType):
    """Yet Another Tiled Tensor Type"""

    def __init__(
        self,
        dtype: DataType,
        layout: Union[TiledTensorLayout, ComposedTensorLayout, TensorLayout],
        scope: Union[DeclareScope, str] = None,
    ):
        if isinstance(scope, str):
            scope = DeclareScope.from_str(scope)
        elif scope is None:
            scope = DeclareScope.Default

        self.dtype = dtype
        self.layout = layout
        self.scope = scope


def tiled_tensor(
    dtype: DataType,
    layout: Union[TiledTensorLayout, ComposedTensorLayout, TensorLayout],
    scope: Union[DeclareScope, str] = None,
):
    assert isinstance(dtype, DataType)
    return TiledTensorType(dtype, layout, scope)


class LogicalEncoding:
    def __init__(self, shape: Tuple[int, ...], layout: TensorLayout):
        self.shape = shape
        self.layout = layout


def logical_encoding(shape: Tuple[int, ...], layout: TensorLayout):
    return LogicalEncoding(shape, layout)
