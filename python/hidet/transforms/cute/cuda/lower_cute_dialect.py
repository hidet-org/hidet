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
from typing import List, Dict, Union, Optional

from hidet.ir.tools import TypeInfer
from hidet.ir.functors import IRRewriter
from hidet.ir.cute.layout import TiledTensorLayout, TensorLayout
from hidet.ir.cute.expr import Op, CallOp
from hidet.ir.cute.type import TiledTensorType

from hidet.ir.func import Function
from hidet.ir.expr import Var, Expr, tensor_var
from hidet.ir.type import BaseType, DataType, PointerType
from hidet.ir.stmt import Stmt, DeclareStmt, EvaluateStmt, SeqStmt, LetStmt, DeclareScope
from hidet.transforms.base import FunctionPass
from hidet.transforms.declare_to_let import DeclareToLetRewriter
from hidet.ir.cute.ops import TiledTensorView, PartitionSrc, PartitionDst, Mask

from .lower_ops import Buffer, emit_op


class LowerCuteDialectRewriter(IRRewriter):
    def __init__(self):
        super().__init__()
        self.stmts: List[Stmt] = []
        self.type_infer = TypeInfer()
        self.var2buffer: Dict[Var, Buffer] = {}

    def alloc_buffer(self, hint: str, op_or_type: Union[Op, TiledTensorType]) -> Buffer:
        if isinstance(op_or_type, Op):
            ttype: TiledTensorType = self.type_infer(CallOp(op_or_type))
        else:
            ttype: TiledTensorType = op_or_type
        layout: Union[TiledTensorLayout, TensorLayout] = ttype.layout
        dtype: DataType = ttype.dtype
        scope: DeclareScope = ttype.scope

        if isinstance(layout, TiledTensorLayout):
            val_layout = layout.val_layout()
            size = val_layout.size()
        else:
            assert isinstance(layout, TensorLayout)
            size = layout.size()

        if isinstance(op_or_type, Op) and isinstance(op_or_type, (TiledTensorView, PartitionSrc, PartitionDst)):
            buf_var: Var = Var(hint, PointerType(base_type=dtype))
        elif scope == DeclareScope.Register:
            buf_var: Var = tensor_var(hint=hint, shape=[size], dtype=dtype)
        else:
            raise NotImplementedError()
        self.append_stmt(DeclareStmt(buf_var))

        buf = Buffer(buf_var=buf_var, dtype=dtype, scope=scope, layout=layout)
        return buf

    def append_stmt(self, stmt: Union[Stmt, Expr]):
        if isinstance(stmt, Expr):
            stmt = EvaluateStmt(stmt)
        self.stmts.append(stmt)

    def flush_stmts(self):
        stmts = self.stmts
        self.stmts = []
        return stmts

    def flatten_stmts(self, stmts: List[Stmt]):
        if len(stmts) == 1:
            return stmts[0]
        else:
            return SeqStmt(stmts)

    def visit_DeclareStmt(self, stmt: DeclareStmt):
        if isinstance(stmt.init, CallOp):
            call = stmt.init
            op = call.op
            buf = self.visit(call)
            self.var2buffer[stmt.var] = buf
            v = self.visit(stmt.var)
            if isinstance(op, (TiledTensorView, PartitionSrc, PartitionDst)):
                assert isinstance(v.type, TiledTensorType)
                var = Var(v.hint, PointerType(base_type=v.type.dtype))
                self.append_stmt(DeclareStmt(var, buf.var, stmt.is_static, stmt.scope))
            elif not isinstance(op, Mask):
                raise NotImplementedError()
            return self.flatten_stmts(self.flush_stmts())
        else:
            v = self.visit(stmt.var)
            init = self.visit(stmt.init) if stmt.init is not None else None
            if v is stmt.var and init is stmt.init:
                return stmt
            else:
                return DeclareStmt(v, init, stmt.is_static, stmt.scope)

    def visit_CallOp(self, call: CallOp):
        args: List[Union[Expr, Buffer]] = []
        for arg in call.op.args:
            arg_type = self.type_infer(arg)
            if isinstance(arg_type, TiledTensorType):
                assert isinstance(arg, Var)
                args.append(self.var2buffer[arg])
            else:
                args.append(self.visit(arg))

        output_ty: BaseType = self.type_infer(call)
        if output_ty.is_void():
            output: Optional[Buffer] = None
        elif isinstance(output_ty, TiledTensorType):
            output: Optional[Buffer] = self.alloc_buffer(call.op.name, call.op)
        else:
            raise NotImplementedError()
        self.append_stmt(emit_op(call.op, args=args, output=output))

        return output

    def visit_LetStmt(self, stmt: LetStmt):
        stmts: List[Stmt] = []
        for bind_var, bind_value in zip(stmt.bind_vars, stmt.bind_values):
            if isinstance(bind_value, CallOp):
                buf = self.visit(bind_value)
                if not isinstance(buf, Buffer):
                    raise NotImplementedError(
                        "The following cute expression has not been lowered to Buffer:\n"
                        + "    {}".format(type(bind_value.op).__name__)
                    )
                self.val2buffer[bind_var] = buf
                buf.var.hint = bind_var.hint
            elif isinstance(bind_value, Var) and isinstance(bind_value.type, TiledTensorType):
                self.memo[bind_var] = bind_value
            else:
                self.append_stmt(DeclareStmt(bind_var, self.visit(bind_value)))
            stmts.extend(self.flush_stmts())
        stmts.append(self.visit(stmt.body))
        if len(stmts) == 1:
            return stmts[0]
        else:
            return SeqStmt(stmts)

    def visit_EvaluateStmt(self, stmt: EvaluateStmt):
        if isinstance(stmt.expr, CallOp):
            ret = self.visit(stmt.expr)
            assert isinstance(ret, Buffer) or ret is None
            stmts = self.flush_stmts()
            if len(stmts) == 1:
                return stmts[0]
            else:
                return SeqStmt(stmts)
        else:
            return super().visit_EvaluateStmt(stmt)


class LowerCuteDialectPass(FunctionPass):
    def process_func(self, func: Function) -> Function:
        return self.apply_transforms(func, [LowerCuteDialectRewriter(), DeclareToLetRewriter()])


def lower_cute_dialect_pass() -> FunctionPass:
    return LowerCuteDialectPass()
