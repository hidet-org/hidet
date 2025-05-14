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
from typing import List, Dict, Union, Optional, Tuple, Type

from hidet.ir.tools import TypeInfer
from hidet.ir.functors import IRRewriter
from hidet.ir.cute import filter
from hidet.ir.cute.layout import TiledTensorLayout, ComposedTensorLayout, TensorLayout
from hidet.ir.cute.expr import Op, CallOp
from hidet.ir.cute.type import TiledTensorType

from hidet.ir.func import Function
from hidet.ir.expr import Var, Expr, tensor_var
from hidet.ir.type import BaseType, DataType
from hidet.ir.stmt import Stmt, AssignStmt, DeclareStmt, EvaluateStmt, SeqStmt, LetStmt, DeclareScope
from hidet.transforms.base import FunctionPass
from hidet.transforms.declare_to_let import DeclareToLetRewriter
from hidet.ir.cute.ops import Tensor, TensorView, Partition, SubTensor, Broadcast, Transpose

from .lower_ops import Buffer, emit_op


class LowerCuteDialectRewriter(IRRewriter):
    """
    A rewriter that rewrites the tile-level primitives into lower-level Hidet IR.
    The translation process is follows:
    1. Each tensor will be translated into a buffer. In the C-like language constructs, a buffer
    corresponds to a pointer to a memory region. We will allocate register memory for the buffer
    when necessary.
    2. The buffer serves as an auxiliary data structure for the translation, and we maintain a
    dictionary mapping variables to buffers. The translation of the tile-level primitives will
    perform computation on the buffers (i.e. pointers).
    3. The translation of the tile-level primitives is done by the emitters defined for each tile-level
    primitive.

    Attributes:
        lower_ops (Tuple[Type[Op], ...]): A tuple of operations to be lowered. Defaults to None.
        stmts (List[Stmt]): A list to store statements.
        type_infer (TypeInfer): An instance of the TypeInfer class for type inference.
        var2buffer (Dict[Var, Buffer]): A dictionary mapping variables to buffers.
    """

    def __init__(self, lower_ops: Tuple[Type[Op], ...] = None):
        """
        Initializes the LowerCuteDialectRewriter.

        Note:
        We currently support lower some operations to Hidet IR. The operations that are not in the
        list will be kept as is.

        Args:
            lower_ops (Tuple[Type[Op], ...], optional): A tuple of operations to be lowered. Defaults to None.
        """
        super().__init__()
        self.stmts: List[Stmt] = []
        self.type_infer = TypeInfer()
        self.var2buffer: Dict[Var, Buffer] = {}
        self.lower_ops = lower_ops

    def _if_lower(self, op: Op):
        """
        Checks if the operation should be lowered.

        Args:
            op (Op): The operation to check.

        Returns:
            bool: True if the operation should be lowered, False otherwise.
        """

        return isinstance(op, self.lower_ops) if self.lower_ops else True

    def alloc_buffer(self, hint: str, op_or_type: Union[Op, TiledTensorType]) -> Buffer:
        """
        Allocates a buffer for a given operation or tiled tensor type.

        Args:
            hint (str): The hint for the buffer.
            op_or_type (Union[Op, TiledTensorType]): The operation or tiled tensor type.

        Returns:
            Buffer: The allocated buffer.
        """
        if isinstance(op_or_type, Op):
            ttype: TiledTensorType = self.type_infer(CallOp(op_or_type))
        else:
            ttype: TiledTensorType = op_or_type
        layout: Union[TiledTensorLayout, ComposedTensorLayout, TensorLayout] = ttype.layout
        dtype: DataType = ttype.dtype
        scope: DeclareScope = ttype.scope

        if isinstance(layout, TiledTensorLayout):
            size = layout.val_count()
        else:
            assert isinstance(layout, (TensorLayout, ComposedTensorLayout))
            size = filter(layout).size()

        if isinstance(op_or_type, Op):
            if isinstance(op_or_type, (TensorView, Partition, SubTensor, Broadcast, Transpose)) or (
                isinstance(op_or_type, Tensor) and scope.is_shared()
            ):
                # no need to allocate buffer
                buffer = None
            elif scope.is_register():
                buffer: Var = tensor_var(hint=hint, shape=[size], dtype=dtype)
            else:
                raise NotImplementedError()
        elif scope.is_register():
            buffer: Var = tensor_var(hint=hint, shape=[size], dtype=dtype)
        else:
            raise NotImplementedError()
        if isinstance(op_or_type, TiledTensorType) or self._if_lower(op_or_type):
            if buffer is not None:
                self.append_stmt(DeclareStmt(buffer))

        # from hidet.ir.expr import TensorType
        # buf_var = ~buf_var[0] if isinstance(buf_var.type, TensorType) else buf_var
        buf = Buffer(buffer=buffer, offset=None, dtype=dtype, scope=scope, layout=layout)
        return buf

    def append_stmt(self, stmt: Union[Stmt, Expr]):
        """
        Appends a statement to the list of statements.

        Args:
            stmt (Union[Stmt, Expr]): The statement or expression to append.
        """

        if isinstance(stmt, Expr):
            stmt = EvaluateStmt(stmt)
        self.stmts.append(stmt)

    def flush_stmts(self):
        """
        Flushes the list of statements.

        Returns:
            List[Stmt]: The flushed statements.
        """

        stmts = self.stmts
        self.stmts = []
        return stmts

    def flatten_stmts(self, stmts: List[Stmt]):
        """
        Flattens a list of statements into a single sequence statement.

        Args:
            stmts (List[Stmt]): The list of statements to flatten.

        Returns:
            Stmt: The flattened statement.
        """

        if len(stmts) == 1:
            return stmts[0]
        else:
            return SeqStmt(stmts)

    def visit_LetStmt(self, stmt: LetStmt):
        """
        Visits a LetStmt node and emit lower-level Hidet IR when applicable.

        Args:
            stmt (LetStmt): The LetStmt node to visit.

        Returns:
            Stmt: The processed statement.
        """

        stmts: List[Stmt] = []
        for bind_var, bind_value in zip(stmt.bind_vars, stmt.bind_values):
            if isinstance(bind_value, CallOp):
                call = bind_value
                op = call.op
                if self._if_lower(op):
                    buf = self.visit(call)
                    if not isinstance(buf, Buffer):
                        raise NotImplementedError(
                            "The following cute expression has not been lowered to Buffer:\n"
                            + "\t{}".format(type(op).__name__)
                        )
                    self.var2buffer[bind_var] = buf
                    buf.buffer.hint = bind_var.hint
                else:
                    v = self.visit(bind_var)
                    ttype: TiledTensorType = self.type_infer(CallOp(op))
                    layout: Union[TiledTensorLayout, TensorLayout] = ttype.layout
                    dtype: DataType = ttype.dtype
                    scope: DeclareScope = ttype.scope
                    buf = Buffer(buffer=v, offset=None, dtype=dtype, scope=scope, layout=layout)
                    self.var2buffer[bind_var] = buf
                    self.append_stmt(DeclareStmt(bind_var, self.visit(call)))
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

    def visit_DeclareStmt(self, stmt: DeclareStmt):
        """
        Visits a DeclareStmt node and emit lower-level Hidet IR when applicable.

        Args:
            stmt (DeclareStmt): The DeclareStmt node to visit.

        Returns:
            Stmt: The processed statement.
        """
        if isinstance(stmt.init, CallOp):
            call = stmt.init
            op = call.op
            if self._if_lower(op):
                buf = self.visit(call)
                self.var2buffer[stmt.var] = buf
                return self.flatten_stmts(self.flush_stmts())
            else:
                v = self.visit(stmt.var)
                init = self.visit(stmt.init)
                self.var2buffer[stmt.var] = v
                if v is stmt.var and init is stmt.init:
                    return stmt
                else:
                    return DeclareStmt(v, init, stmt.is_static, stmt.scope)
        return super().visit_DeclareStmt(stmt)

    # TODO
    def visit_AssignStmt(self, stmt: AssignStmt):
        """
        Visits an AssignStmt node and emit lower-level Hidet IR when applicable.

        Args:
            stmt (AssignStmt): The AssignStmt node to visit.

        Returns:
            Stmt: The processed statement.
        """

        if isinstance(stmt.value, CallOp):
            call = stmt.value
            op = call.op
            if self._if_lower(op):
                args: List[Union[Expr, Buffer]] = []
                for arg in op.args:
                    if isinstance(arg, (tuple, list)):
                        args.append(tuple(self.visit(v) for v in arg))
                    else:
                        arg_type = self.type_infer(arg)
                        if isinstance(arg_type, TiledTensorType) and isinstance(arg, Var):
                            args.append(self.var2buffer[arg])
                        else:
                            args.append(self.visit(arg))
                output = self.var2buffer[stmt.var]
                self.append_stmt(emit_op(call.op, args=args, output=output))
                return self.flatten_stmts(self.flush_stmts())
        return super().visit_AssignStmt(stmt)

    def visit_CallOp(self, call: CallOp):
        """
        Visits a CallOp and emit the corresponding lower-level IR.

        Args:
            call (CallOp): The CallOp to visit.

        Returns:
            Union[Buffer, None]: The result of processing the CallOp.
        """

        if self._if_lower(call.op):
            args: List[Union[Expr, Buffer]] = []
            for arg in call.op.args:
                if arg is None:  # optional argument in copy operation
                    args.append(None)
                elif isinstance(arg, (tuple, list)):
                    args.append(tuple(self.visit(v) for v in arg))
                else:
                    arg_type = self.type_infer(arg)
                    if isinstance(arg_type, TiledTensorType) and isinstance(arg, Var):
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
        else:
            args = [self.visit(arg) for arg in call.op.args]
            op = call.op.reforward(args)
            if op is call.op:
                return call
            else:
                return op.make_call()

    def visit_EvaluateStmt(self, stmt: EvaluateStmt):
        """
        Visits an EvaluateStmt and emit lower-level Hidet IR when applicable.

        Args:
            stmt (EvaluateStmt): The EvaluateStmt to visit.

        Returns:
            Stmt: The processed statement.
        """
        if isinstance(stmt.expr, CallOp):
            call = stmt.expr
            op = call.op
            if self._if_lower(op):
                ret = self.visit(stmt.expr)
                assert isinstance(ret, Buffer) or ret is None
                stmts = self.flush_stmts()
                if len(stmts) == 1:
                    return stmts[0]
                else:
                    return SeqStmt(stmts)
        return super().visit_EvaluateStmt(stmt)


class LowerCuteDialectPass(FunctionPass):
    def __init__(self, lower_ops: Tuple[Type[Op], ...] = None):
        super().__init__()
        self.lower_ops = lower_ops

    def process_func(self, func: Function) -> Function:
        return self.apply_transforms(func, [LowerCuteDialectRewriter(self.lower_ops), DeclareToLetRewriter()])


def lower_cute_dialect_pass(lower_ops: Tuple[Type[Op], ...] = None) -> FunctionPass:
    return LowerCuteDialectPass(lower_ops)
