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
from typing import List, Dict, Union
from hidet.ir.node import Node
from hidet.ir.type import BaseType
from hidet.ir.expr import Expr
from hidet.ir.cute import TiledTensorLayout, TensorLayout
from hidet.ir.cute.algorithm import TiledCopy

_ScalarConst = Union[str, int, float, bool, BaseType, TiledTensorLayout, TensorLayout, TiledCopy]
CConst = Union[_ScalarConst, List[_ScalarConst]]


class Op(Node):
    def __init__(self, args: List[Expr] = None, attrs: Dict[str, CConst] = None):
        self.args: List[Expr] = args if args is not None else []
        self.attrs: Dict[str, CConst] = attrs if attrs is not None else {}

    @classmethod
    def op_name(cls):
        # camel to snake (e.g., CamelName -> camel_name)
        camel_name = cls.__name__
        snake_name = "".join(["_" + c.lower() if c.isupper() else c for c in camel_name]).lstrip("_")
        return snake_name

    @property
    def name(self):
        return self.op_name()

    def make_call(self):
        return CallOp(self)

    def write_memory_op(self) -> bool:
        return False

    def infer_type(self, arg_types: List[BaseType]) -> BaseType:
        raise NotImplementedError(
            "'infer_type' method has not been implementd for the following operator: \n{}".format(type(self).__name__)
        )


class CallOp(Expr):
    def __init__(self, op: Op):
        self.op: Op = op


def call(op: Op):
    return CallOp(op)
