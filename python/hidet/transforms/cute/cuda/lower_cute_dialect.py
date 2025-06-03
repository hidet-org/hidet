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
from hidet.ir.functors import IRRewriter, IRVisitor
from hidet.ir.cute import filter
from hidet.ir.cute.layout import TiledTensorLayout, ComposedTensorLayout, TensorLayout
from hidet.ir.cute.expr import Op, CallOp
from hidet.ir.cute.type import TiledTensorType

from hidet.ir.func import Function
from hidet.ir.module import IRModule
from hidet.ir.expr import Var, Expr, tensor_var
from hidet.ir.type import BaseType, DataType, OpaqueType, tensor_type
from hidet.ir.primitives.cuda.tensor_map import create_tensor_map
from hidet.ir.dtypes import u64, u32, boolean
from hidet.ir.stmt import (
    IfStmt,
    LaunchKernelStmt,
    Stmt,
    AssignStmt,
    DeclareStmt,
    EvaluateStmt,
    SeqStmt,
    LetStmt,
    DeclareScope,
    BufferStoreStmt,
)
from hidet.transforms.base import Pass
from hidet.transforms.declare_to_let import DeclareToLetRewriter
from hidet.ir.cute.ops import Tensor, TensorView, Partition, SubTensor, Broadcast, Transpose, MBarriers

from hidet.transforms.lower_integer_subbyte import is_pointer_type, get_pointer_base_type
from .lower_ops import Buffer, emit_op, TmaTensor


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
        self.func_params: List[Var] = []

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
            if isinstance(op_or_type, (MBarriers, TensorView, Partition, SubTensor, Broadcast, Transpose)) or (
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

            if isinstance(call.op, TensorView):
                tensor = call.op
                annotations = tensor.annotations
                if "tma_tensors" in annotations:
                    tma_tensors = annotations["tma_tensors"]
                    for _ in tma_tensors:
                        tensor_map = Var("tensor_map", OpaqueType("CUtensorMap", "const", "__grid_constant__"))
                        self.func_params.append(tensor_map)
                        output.tensor_maps.append(tensor_map)
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

    def visit_IfStmt(self, stmt: IfStmt):
        cond_ty = self.type_infer(stmt.cond)
        if isinstance(cond_ty, TiledTensorType):
            assert isinstance(stmt.cond, Var)
            cond_buffer = self.var2buffer[stmt.cond]
            if isinstance(cond_buffer, Buffer):
                then_body = self.visit(stmt.then_body)
                else_body = self.visit(stmt.else_body) if stmt.else_body is not None else None
                assert cond_buffer.dtype.is_boolean() or cond_buffer.dtype.is_integer()
                assert cond_buffer.layout.size() == 1
                new_cond = boolean(cond_buffer.buffer[0])
                return IfStmt(new_cond, then_body, else_body)
        return super().visit_IfStmt(stmt)

    def visit_Function(self, func: Function):
        self.func_params = func.params
        body = self.visit(func.body)
        return Function(func.name, self.func_params, body, func.ret_type, func.kind, func.attrs)


class CollectTmaTensors(IRVisitor):
    def __init__(self):
        super().__init__()
        self.tma_tensors = []

    def visit_TensorView(self, e: TensorView):
        annotations = e.annotations
        if "tma_tensors" in annotations:
            tma_tensors = annotations["tma_tensors"]
            self.tma_tensors.extend(tma_tensors)

    def collect(self, func: Function):
        self.visit(func)

        return self.tma_tensors


class TmaTensorExtractor(IRVisitor):
    def __init__(self, func_name2tma_tensors: Dict[str, List[TmaTensor]]):
        super().__init__()
        self.func_name2tma_tensors = func_name2tma_tensors
        self.tma_tensors: List[TmaTensor] = []

    def visit_LaunchKernelStmt(self, stmt: LaunchKernelStmt):
        func_name = stmt.func_var.name
        if func_name in self.func_name2tma_tensors:
            tma_tensors = self.func_name2tma_tensors[func_name]
            for tma_tensor in tma_tensors:
                tma_tensor.set_base_pointer(stmt.args[tma_tensor.param_idx])
            self.tma_tensors.extend(tma_tensors)

    def extract(self, func: Function):
        self.visit(func)
        return self.tma_tensors


class TmaTensorRewriter(IRRewriter):
    def __init__(self, tma_tensors: List[TmaTensor], func_name2tma_tensors: Dict[str, List[TmaTensor]]):
        super().__init__()
        self.infer_type = TypeInfer()
        self.tma_tensors = tma_tensors
        self.func_name2tma_tensors = func_name2tma_tensors
        self.stmts: List[Stmt] = []
        self.tma_tensor2tensor_map: Dict[TmaTensor, Var] = {}

    def declare_var(self, v: Var = None, hint: str = None, e: Expr = None):
        if v is not None:
            if e is not None:
                self.stmts.append(DeclareStmt(v, e))
            else:
                self.stmts.append(DeclareStmt(v))
            return v
        else:
            v_ty = self.infer_type(e)
            v = Var(hint, v_ty)
            self.stmts.append(DeclareStmt(v, e))
            return v

    def assign(self, lhs: Var, rhs: Expr):
        self.stmts.append(AssignStmt(lhs, rhs))

    def buffer_store(self, buf: Var, indices: List[Expr], value: Expr):
        self.stmts.append(BufferStoreStmt(buf, indices, value))

    def append_stmt(self, stmt: Union[Stmt, Expr]):
        if isinstance(stmt, Expr):
            stmt = EvaluateStmt(stmt)
        self.stmts.append(stmt)

    def flush_stmts(self):
        stmts = self.stmts
        self.stmts = []
        return stmts

    def visit_LaunchKernelStmt(self, stmt: LaunchKernelStmt):
        func_name = stmt.func_var.name
        if func_name in self.func_name2tma_tensors:
            func_var = self.visit(stmt.func_var)
            tma_tensors = self.func_name2tma_tensors[func_name]
            args = [self.visit(arg) for arg in stmt.args]
            args += [self.tma_tensor2tensor_map[tma_tensor] for tma_tensor in tma_tensors]
            grid_dim = self.visit(stmt.grid_dim)
            cluster_dim = self.visit(stmt.cluster_dim)
            block_dim = self.visit(stmt.block_dim)
            shared_mem_bytes = self.visit(stmt.shared_mem_bytes)
            target = self.visit(stmt.target)
            return LaunchKernelStmt(func_var, args, grid_dim, cluster_dim, block_dim, shared_mem_bytes, target)
        return super().visit_LaunchKernelStmt(stmt)

    def visit_Function(self, func: Function):
        tensor_map_ty = OpaqueType("CUtensorMap")
        for tma_tensor in self.tma_tensors:
            rank = tma_tensor.dim
            swizzle = tma_tensor.swizzle
            base_pointer = tma_tensor.get_base_pointer()
            base_pointer_ty = self.infer_type(base_pointer)
            assert is_pointer_type(base_pointer_ty)
            data_type = get_pointer_base_type(base_pointer_ty)
            if data_type.is_integer_subbyte():
                divisor = data_type.storage.nbits // data_type.nbits
                data_type = data_type.storage
            else:
                divisor = 1
            tensor_map = self.declare_var(Var("tensor_map", tensor_map_ty))
            size = self.declare_var(Var("size", tensor_type(u64, [rank])))
            for i in range(rank):
                extent = tma_tensor.extents[i]
                extent = extent // divisor if i == 0 else extent
                self.buffer_store(size, [i], extent)
            stride = self.declare_var(Var("stride", tensor_type(u64, [rank - 1])))
            for i in range(rank - 1):
                self.buffer_store(stride, [i], (tma_tensor.strides[i + 1] // divisor) * data_type.nbytes)
            box_size = self.declare_var(Var("box_size", tensor_type(u32, [rank])))
            for i in range(rank):
                box_shape = tma_tensor.box_shape[i]
                box_shape = box_shape // divisor if i == 0 else box_shape
                self.buffer_store(box_size, [i], box_shape)
            elem_stride = self.declare_var(Var("elem_stride", tensor_type(u32, [rank])))
            for i in range(rank):
                self.buffer_store(elem_stride, [i], u32(1))
            self.append_stmt(
                create_tensor_map(
                    ~tensor_map,
                    data_type.name,
                    rank,
                    base_pointer,
                    size,
                    stride,
                    box_size,
                    elem_stride,
                    swizzle=swizzle,
                )
            )
            self.tma_tensor2tensor_map[tma_tensor] = tensor_map
        stmts = self.flush_stmts()
        body = self.visit(func.body)
        if isinstance(body, SeqStmt):
            body = stmts + list(body.seq)
        else:
            body = stmts + [body]
        if len(body) == 1:
            body = body[0]
        else:
            body = SeqStmt(body)

        return Function(func.name, func.params, body, func.ret_type, func.kind, func.attrs)


class LowerCuteDialectPass(Pass):
    def __init__(self, lower_ops: Tuple[Type[Op], ...] = None):
        super().__init__()
        self.lower_ops = lower_ops

    def process_module(self, ir_module: IRModule) -> IRModule:
        # collect tma tensors
        func_name2tma_tensors = {}
        for name, func in ir_module.functions.items():
            tma_tensors = CollectTmaTensors().collect(func)
            if len(tma_tensors) > 0:
                func_name2tma_tensors[name] = tma_tensors

        new_funcs = {}
        for name, func in ir_module.functions.items():
            if name.startswith("launch") and func.kind == "public":
                tma_tensors = TmaTensorExtractor(func_name2tma_tensors).extract(func)
                rewriter = TmaTensorRewriter(tma_tensors, func_name2tma_tensors)
                func = rewriter(func)
                new_funcs[name] = func

        for name, func in ir_module.functions.items():
            func = new_funcs.get(name, func)
            lower_cute_rewriter = LowerCuteDialectRewriter(self.lower_ops)
            func = lower_cute_rewriter(func)
            declare_to_let = DeclareToLetRewriter()
            func = declare_to_let(func)
            new_funcs[name] = func
        return ir_module.copy().reset_funcs(new_funcs, ir_module.global_vars)


def lower_cute_dialect_pass(lower_ops: Tuple[Type[Op], ...] = None) -> Pass:
    return LowerCuteDialectPass(lower_ops)
