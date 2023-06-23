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
from typing import Union, Optional, Sequence
from hidet.ir.type import BaseType, DataType, tensor_type, tensor_pointer_type
from hidet.ir.layout import DataLayout
from hidet.ir.expr import Expr, cast
from hidet.ir.stmt import DeclareScope


class Declaration:
    def __init__(self, scope, tp, init: Optional[Expr] = None, is_static=False):
        self.scope: DeclareScope = scope
        self.type: Optional[BaseType] = tp
        self.init: Optional[Expr] = init
        self.is_static: bool = is_static


# class TypeDecorator:
#     def __init__(self, decorated_type: BaseType, scope: Optional[Union[str, DeclareScope]], is_static: bool = False):
#         if isinstance(scope, str):
#             scope = DeclareScope.from_str(scope)
#         self.decorated_type: BaseType = decorated_type
#         self.scope = scope
#         self.is_static = is_static


def tensor(
    dtype: Union[DataType, str],
    shape: Optional[Sequence[Union[Expr, int]]] = None,
    layout: Optional[DataLayout] = None,
    scope: Union[DeclareScope, str] = DeclareScope.Default,
    is_static: bool = False,
):
    if isinstance(scope, str):
        scope = DeclareScope.from_str(scope)
    return Declaration(scope, tp=tensor_type(dtype, shape, layout), init=None, is_static=is_static)


def tensor_pointer(
    dtype: Union[DataType, str],
    shape: Optional[Sequence[Union[Expr, int]]] = None,
    layout: Optional[DataLayout] = None,
    init: Optional[Expr] = None,
):
    return Declaration(scope=DeclareScope.Default, tp=tensor_pointer_type(dtype, shape, layout), init=init)


def as_tensor_pointer(
    expr: Expr,
    dtype: Union[DataType, str],
    shape: Optional[Sequence[Union[Expr, int]]] = None,
    layout: Optional[DataLayout] = None,
) -> Expr:
    return cast(expr, tensor_pointer_type(dtype, shape, layout))


def shared_tensor(
    dtype: Union[DataType, str], shape: Optional[Sequence[Union[Expr, int]]] = None, layout: Optional[DataLayout] = None
):
    return tensor(scope=DeclareScope.Shared, dtype=dtype, shape=shape, layout=layout)


def register_tensor(
    dtype: Union[DataType, str], shape: Optional[Sequence[Union[Expr, int]]] = None, layout: Optional[DataLayout] = None
):
    return tensor(scope=DeclareScope.Register, dtype=dtype, shape=shape, layout=layout)
