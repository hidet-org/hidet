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
from typing import Dict, Type, List, Union, Tuple, Optional, cast
import inspect
from hidet.ir.expr import Expr, Var, var
from hidet.ir.stmt import DeclareScope
from hidet.ir.type import DataType
from hidet.ir.dtypes import i32
from hidet.ir.stmt import Stmt, ForMappingStmt

from hidet.ir.cute.expr import Op
from hidet.ir.cute.int_tuple import is_integer
from hidet.ir.cute.layout import TiledTensorLayout, ComposedTensorLayout, TensorLayout

from hidet.ir.builders import StmtBuilder
from hidet.ir.builders.stmt_builder import StmtScope
from hidet.ir.mapping import repeat_map
from hidet.ir.tools import infer_type


class Buffer:
    def __init__(
        self,
        buffer: Var,
        offset: Expr,
        dtype: DataType,
        layout: Union[TiledTensorLayout, ComposedTensorLayout, TensorLayout],
        scope: Union[DeclareScope, str] = None,
    ):
        if isinstance(scope, str):
            scope = DeclareScope.from_str(scope)
        elif scope is None:
            scope = DeclareScope.Default
        self.buffer: Var = buffer
        self.offset: Expr = offset
        self.dtype: DataType = dtype
        self.layout: Union[TiledTensorLayout, ComposedTensorLayout, TensorLayout] = layout
        self.scope = scope


class OpEmitter(StmtBuilder):
    def auto_var(self, v: Var = None, hint: str = None, e: Expr = None):
        if v is not None:
            self.declare(v)
            return v
        v_ty = infer_type(e)
        v = var(hint, v_ty)
        self.declare(v, e)
        return v

    def for_grid(self, shape) -> StmtScope:
        if isinstance(shape, tuple):
            shape = list(shape)
        elif is_integer(shape):
            shape = [shape]
        iter_names = self._name_index_vars(len(shape))
        iter_vars = [var(name) for name in iter_names]
        ret_vars = iter_vars[0] if len(iter_vars) == 1 else tuple(iter_vars)
        mapping = repeat_map(shape)
        return StmtScope(self, stmts=ForMappingStmt(iter_vars, mapping, i32(0), cast(Stmt, None)), ret=ret_vars)

    def get_smem_ptr(self, op: Op, dtype: DataType, nbytes: int) -> Expr:
        from hidet.ir.primitives.cuda.smem import dynamic_shared_memory

        requested: int = self.request_smem_nbytes(op)
        if requested == 0:
            raise RuntimeError(
                'Please implement the "request_smem_nbyts" method to return a positive integer'
                'before accessing the requested shared_memory.'
            )
        if nbytes > requested:
            raise RuntimeError(f"Requested {nbytes} bytes of shared memory, but only {requested} bytes are allocated.")
        smem_offset = "smem_offset"
        if smem_offset not in op.annotations:
            return dynamic_shared_memory(0, dtype=dtype)
            # FIXME:
            # raise RuntimeError(
            #     "Missing shared memory offset annotation. This usually indicates a potential compile error."
            # )

        return dynamic_shared_memory(op.annotations[smem_offset], dtype=dtype)

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


def request_smem_nbytes(op: Op) -> int:
    emitter_cls = get_op_emitter(op)
    emitter = emitter_cls()
    return emitter.request_smem_nbytes(op)
