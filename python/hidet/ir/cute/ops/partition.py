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
from typing import List

from hidet.ir.type import BaseType
from hidet.ir.expr import Expr
from hidet.ir.stmt import DeclareScope

from hidet.ir.cute.expr import Op
from hidet.ir.cute.algorithm import TiledCopy
from hidet.ir.cute.type import tiled_tensor, TiledTensorType
from hidet.ir.cute import TensorLayout, composition, make_layout
from hidet.ir.cute.int_tuple import prefix_product


class PartitionSrc(Op):
    def __init__(self, x: Expr, tiled_copy: TiledCopy):
        super().__init__(args=[x], attrs={"tiled_copy": tiled_copy})
        self.x: Expr = x
        self.tiled_copy: TiledCopy = tiled_copy

    def infer_type(self, arg_types: List[BaseType]) -> BaseType:
        x_type = arg_types[0]
        if not isinstance(x_type, TiledTensorType):
            raise TypeError(f"Type mismatch.(got:x({x_type}))")
        shape, thrval_layout = self.tiled_copy.src_tv_layout()
        if x_type.scope == DeclareScope.Register:
            shape = (thrval_layout[0][1].shape, thrval_layout[1].shape)
            stride = prefix_product(shape)
            return tiled_tensor(x_type.dtype, TensorLayout(shape, stride), x_type.scope)

        return tiled_tensor(
            x_type.dtype, composition(x_type.layout, make_layout(thrval_layout[0][1], thrval_layout[1])), x_type.scope
        )


def partition_src(x: Expr, tiled_copy: TiledCopy):
    return PartitionSrc(x, tiled_copy).make_call()


class PartitionDst(Op):
    def __init__(self, x: Expr, tiled_copy: TiledCopy):
        super().__init__(args=[x], attrs={"tiled_copy": tiled_copy})
        self.x: Expr = x
        self.tiled_copy: TiledCopy = tiled_copy

    def infer_type(self, arg_types: List[BaseType]) -> BaseType:
        x_type = arg_types[0]
        if not isinstance(x_type, TiledTensorType):
            raise TypeError(f"Type mismatch.(got:x({x_type}))")
        shape, thrval_layout = self.tiled_copy.dst_tv_layout()
        if x_type.scope == DeclareScope.Register:
            shape = (thrval_layout[0][1].shape, thrval_layout[1].shape)
            stride = prefix_product(shape)
            return tiled_tensor(x_type.dtype, TensorLayout(shape, stride), x_type.scope)

        return tiled_tensor(
            x_type.dtype, composition(x_type.layout, make_layout(thrval_layout[0][1], thrval_layout[1])), x_type.scope
        )


def partition_dst(x: Expr, tiled_copy: TiledCopy):
    return PartitionDst(x, tiled_copy).make_call()
