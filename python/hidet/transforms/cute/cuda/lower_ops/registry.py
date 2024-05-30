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
from typing import Dict, Type, List, Union, Tuple, Optional
import inspect
from hidet.ir.expr import Expr, Var
from hidet.ir.stmt import DeclareScope
from hidet.ir.type import DataType
from hidet.ir.stmt import Stmt, AssignStmt

from hidet.ir.cute.expr import Op
from hidet.ir.cute.layout import TiledTensorLayout, TensorLayout

from hidet.ir.builders import StmtBuilder


class Buffer:
    def __init__(
        self,
        buf_var: Var,
        dtype: DataType,
        layout: Union[TiledTensorLayout, TensorLayout],
        scope: Union[DeclareScope, str] = None,
    ):
        if isinstance(scope, str):
            scope = DeclareScope.from_str(scope)
        elif scope is None:
            scope = DeclareScope.Default
        self.var: Var = buf_var
        self.dtype: DataType = dtype
        self.layout: Union[TiledTensorLayout, TensorLayout] = layout
        self.scope = scope


class OpEmitter(StmtBuilder):
    def assign(self, var: Expr, value: Expr):
        self.append(AssignStmt(var, value))

    def get_smem_ptr(self, op: Op, nbytes: int) -> Expr:
        from hidet.ir.primitives.cuda.smem import dynamic_shared_memory

        requested: int = self.request_smem_nbytes(op)
        if requested == 0:
            raise RuntimeError(
                'Please implement the "request_smem_nbyts" method to return a positive integer'
                'before accessing the requested shared_memory.'
            )

        if nbytes > requested:
            raise RuntimeError(f"Requested {nbytes} bytes of shared memory, but only {requested} bytes are allocated.")

        return dynamic_shared_memory(0, 'uint8_t')

    def request_smem_nbytes(self, op: Op) -> int:
        return 0

    def emit(self, op: Op, args: List[Union[Buffer, Expr]], output: Optional[Buffer]):
        raise NotImplementedError()


_registered_implementations: Dict[Type[Op], Type[OpEmitter]] = {}


def register_impl(op_cls: Type[Op]):
    def decorator(emitter_cls: Type[OpEmitter]):
        _registered_implementations[op_cls] = emitter_cls

    return decorator


def get_op_emitter(op_or_op_cls: Union[Op, Type[Op]]) -> Type[OpEmitter]:
    if isinstance(op_or_op_cls, Op):
        op_cls = type(op_or_op_cls)
    elif issubclass(op_or_op_cls, Op):
        op_cls = op_or_op_cls
    else:
        raise RuntimeError(f"Cannot get op emitter for {op_or_op_cls}")

    if op_cls not in _registered_implementations:
        parent_classes: Tuple = inspect.getmro(op_cls)
        for cls in parent_classes:
            if cls in _registered_implementations:
                _registered_implementations[op_cls] = _registered_implementations[cls]
                break
        else:
            raise RuntimeError(f"Cannot get op emitter for {op_cls.op_name()}")
    emitter_cls = _registered_implementations[op_cls]

    return emitter_cls


def emit_op(op: Op, args: List[Buffer], output: Buffer) -> Stmt:
    emitter_cls = get_op_emitter(op)
    emitter = emitter_cls()
    emitter.emit(op, args, output)
    return emitter.finish()
