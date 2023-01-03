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
# pylint: disable=bad-staticmethod-argument
from typing import Any, Union, Mapping, Sequence, Type, List
from hidet.ir.node import Node
from hidet.ir.type import DataType, TensorType, PointerType, TensorPointerType, ReferenceType, VoidType
from hidet.ir.expr import (
    Add,
    Sub,
    Multiply,
    Div,
    Mod,
    FloorDiv,
    Neg,
    LessThan,
    LessEqual,
    Equal,
    NotEqual,
    LogicalAnd,
    LogicalOr,
)
from hidet.ir.expr import (
    LogicalNot,
    BitwiseAnd,
    BitwiseOr,
    BitwiseNot,
    BitwiseXor,
    LeftShift,
    RightShift,
    TensorElement,
)
from hidet.ir.expr import TensorSlice, IfThenElse, Call, Let, Var, Constant, Cast, Dereference, Address, Reference
from hidet.ir.expr import BinaryOp, Expr
from hidet.ir.stmt import EvaluateStmt, DeclareStmt, BufferStoreStmt, AssignStmt, LetStmt, ForStmt, ForTaskStmt, SeqStmt
from hidet.ir.stmt import WhileStmt, BreakStmt, ContinueStmt, IfStmt, ReturnStmt, AsmStmt, AssertStmt, BlackBoxStmt
from hidet.ir.stmt import LaunchKernelStmt
from hidet.ir.func import Function
from hidet.ir.compute import TensorNode, ScalarNode, ReduceCompute, ArgReduceCompute, GridCompute
from hidet.ir.dialects.pattern import AnyExpr

# todo: current implementation of method dispatch is not very human readable, will consider refactoring to
#       use if-elif-else chain


class NodeFunctor:
    def __init__(self, use_memo=True):
        self.memo = {} if use_memo else None
        if not hasattr(self.__class__, 'dispatch_table'):
            self.setup_dispatch_table()

    def __call__(self, node: Any):
        return self.visit(node)

    def visit(self, node: Union[Node, tuple, list]):
        key = id(node) if isinstance(node, list) else node
        if self.memo is not None and key in self.memo:
            return self.memo[key]
        if isinstance(node, Node):
            idx = node.class_index() if node is not None else 0
            # noinspection PyUnresolvedReferences
            dispatch_table = self.__class__.dispatch_table  # pylint: disable=no-member
            if idx >= len(dispatch_table):
                raise NotImplementedError(
                    'Does not implement dispatch function in "{}" for node "{}"'.format(
                        type(self).__qualname__, type(node).__qualname__
                    )
                )
            ret = dispatch_table[idx](self, node)
        elif isinstance(node, tuple):
            ret = tuple(self.visit(v) for v in node)
        elif isinstance(node, list):
            ret = [self.visit(v) for v in node]
        else:
            raise NotImplementedError("Can not dispatch object with type {}".format(type(node)))
        if self.memo is not None:
            self.memo[key] = ret
        return ret

    @staticmethod
    def get_dispatch_mapping(cls) -> Mapping[Type[Node], Any]:  # pylint: disable=unused-argument
        return {}

    @classmethod
    def setup_dispatch_table(cls: Type[Node]):
        cls_stack: List[type] = [cls]
        mapping = {}
        while len(cls_stack) > 0:
            cur_cls = cls_stack.pop()
            if hasattr(cur_cls, 'get_dispatch_mapping'):
                cur_mapping = cur_cls.get_dispatch_mapping(cls)
                for k, v in cur_mapping.items():
                    if k not in mapping:
                        mapping[k] = v
            cls_stack.extend(cur_cls.__bases__)
        setattr(cls, 'dispatch_table', Node.dispatch_table(mapping))


class ExprFunctor(NodeFunctor):
    @staticmethod
    def get_dispatch_mapping(cls) -> Mapping[Type[Node], Any]:
        return {
            Add: cls.visit_Add,
            Sub: cls.visit_Sub,
            Multiply: cls.visit_Multiply,
            Div: cls.visit_Div,
            Mod: cls.visit_Mod,
            FloorDiv: cls.visit_FloorDiv,
            Neg: cls.visit_Neg,
            LessThan: cls.visit_LessThan,
            LessEqual: cls.visit_LessEqual,
            Equal: cls.visit_Equal,
            NotEqual: cls.visit_NotEqual,
            LogicalAnd: cls.visit_And,
            LogicalOr: cls.visit_Or,
            LogicalNot: cls.visit_Not,
            BitwiseAnd: cls.visit_BitwiseAnd,
            BitwiseOr: cls.visit_BitwiseOr,
            BitwiseNot: cls.visit_BitwiseNot,
            BitwiseXor: cls.visit_BitwiseXor,
            LeftShift: cls.visit_LeftShift,
            RightShift: cls.visit_RightShift,
            TensorElement: cls.visit_TensorElement,
            TensorSlice: cls.visit_TensorSlice,
            IfThenElse: cls.visit_IfThenElse,
            Call: cls.visit_Call,
            Let: cls.visit_Let,
            Var: cls.visit_Var,
            Constant: cls.visit_Constant,
            Cast: cls.visit_Cast,
            Dereference: cls.visit_Dereference,
            Address: cls.visit_Address,
            Reference: cls.visit_Reference,
            TensorNode: cls.visit_TensorNode,
            ScalarNode: cls.visit_ScalarNode,
            AnyExpr: cls.visit_AnyExpr,
        }

    def visit_Add(self, e: Add):
        raise NotImplementedError()

    def visit_Sub(self, e: Sub):
        raise NotImplementedError()

    def visit_Multiply(self, e: Multiply):
        raise NotImplementedError()

    def visit_Div(self, e: Div):
        raise NotImplementedError()

    def visit_Mod(self, e: Mod):
        raise NotImplementedError()

    def visit_FloorDiv(self, e: FloorDiv):
        raise NotImplementedError()

    def visit_LessThan(self, e: LessThan):
        raise NotImplementedError()

    def visit_LessEqual(self, e: LessEqual):
        raise NotImplementedError()

    def visit_Equal(self, e: Equal):
        raise NotImplementedError()

    def visit_NotEqual(self, e: NotEqual):
        raise NotImplementedError()

    def visit_And(self, e: LogicalAnd):
        raise NotImplementedError()

    def visit_Or(self, e: LogicalOr):
        raise NotImplementedError()

    def visit_Neg(self, e: Neg):
        raise NotImplementedError()

    def visit_Not(self, e: LogicalNot):
        raise NotImplementedError()

    def visit_BitwiseAnd(self, e: BitwiseAnd):
        raise NotImplementedError()

    def visit_BitwiseOr(self, e: BitwiseOr):
        raise NotImplementedError()

    def visit_BitwiseNot(self, e: BitwiseNot):
        raise NotImplementedError()

    def visit_BitwiseXor(self, e: BitwiseXor):
        raise NotImplementedError()

    def visit_LeftShift(self, e: LeftShift):
        raise NotImplementedError()

    def visit_RightShift(self, e: RightShift):
        raise NotImplementedError()

    def visit_TensorElement(self, e: TensorElement):
        raise NotImplementedError()

    def visit_TensorSlice(self, e: TensorSlice):
        raise NotImplementedError()

    def visit_IfThenElse(self, e: IfThenElse):
        raise NotImplementedError()

    def visit_Cast(self, e: Cast):
        raise NotImplementedError()

    def visit_Dereference(self, e: Dereference):
        raise NotImplementedError()

    def visit_Address(self, e: Address):
        raise NotImplementedError()

    def visit_Reference(self, e: Reference):
        raise NotImplementedError()

    def visit_Call(self, e: Call):
        raise NotImplementedError()

    def visit_Let(self, e: Let):
        raise NotImplementedError()

    def visit_Var(self, e: Var):
        raise NotImplementedError()

    def visit_Constant(self, e: Constant):
        raise NotImplementedError()

    def visit_ScalarNode(self, e: ScalarNode):
        raise NotImplementedError()

    def visit_TensorNode(self, e: TensorNode):
        raise NotImplementedError()

    def visit_AnyExpr(self, e: AnyExpr):
        raise NotImplementedError()


class ExprVisitor(ExprFunctor):
    def visit_Add(self, e: Add):
        self.visit(e.a)
        self.visit(e.b)

    def visit_Sub(self, e: Sub):
        self.visit(e.a)
        self.visit(e.b)

    def visit_Multiply(self, e: Multiply):
        self.visit(e.a)
        self.visit(e.b)

    def visit_Div(self, e: Div):
        self.visit(e.a)
        self.visit(e.b)

    def visit_Mod(self, e: Mod):
        self.visit(e.a)
        self.visit(e.b)

    def visit_FloorDiv(self, e: FloorDiv):
        self.visit(e.a)
        self.visit(e.b)

    def visit_LessThan(self, e: LessThan):
        self.visit(e.a)
        self.visit(e.b)

    def visit_LessEqual(self, e: LessEqual):
        self.visit(e.a)
        self.visit(e.b)

    def visit_Equal(self, e: Equal):
        self.visit(e.a)
        self.visit(e.b)

    def visit_NotEqual(self, e: NotEqual):
        self.visit(e.a)
        self.visit(e.b)

    def visit_And(self, e: LogicalAnd):
        self.visit(e.a)
        self.visit(e.b)

    def visit_Or(self, e: LogicalOr):
        self.visit(e.a)
        self.visit(e.b)

    def visit_Neg(self, e: Neg):
        self.visit(e.a)

    def visit_Not(self, e: LogicalNot):
        self.visit(e.a)

    def visit_BitwiseAnd(self, e: BitwiseAnd):
        self.visit(e.a)
        self.visit(e.b)

    def visit_BitwiseOr(self, e: BitwiseOr):
        self.visit(e.a)
        self.visit(e.b)

    def visit_BitwiseXor(self, e: BitwiseXor):
        self.visit(e.a)
        self.visit(e.b)

    def visit_BitwiseNot(self, e: BitwiseNot):
        self.visit(e.base)

    def visit_LeftShift(self, e: LeftShift):
        self.visit(e.base)
        self.visit(e.cnt)

    def visit_RightShift(self, e: RightShift):
        self.visit(e.base)
        self.visit(e.cnt)

    def visit_TensorElement(self, e: TensorElement):
        self.visit(e.base)
        for idx in e.indices:
            self.visit(idx)

    def visit_TensorSlice(self, e: TensorSlice):
        self.visit(e.base)
        for idx, start, end in zip(e.starts, e.indices, e.ends):
            for obj in [idx, start, end]:
                if obj is not None:
                    self.visit(obj)

    def visit_IfThenElse(self, e: IfThenElse):
        self.visit(e.cond)
        self.visit(e.then_expr)
        self.visit(e.else_expr)

    def visit_Call(self, e: Call):
        self.visit(e.func_var)
        for arg in e.args:
            self.visit(arg)

    def visit_Let(self, e: Let):
        self.visit(e.value)
        self.visit(e.var)
        self.visit(e.body)

    def visit_Var(self, e: Var):
        pass

    def visit_Constant(self, e: Constant):
        pass

    # compute dialect
    def visit_ScalarNode(self, e: ScalarNode):
        if e.scalar_compute:
            if isinstance(e.scalar_compute, ReduceCompute):
                self.visit(e.scalar_compute.value)
            elif isinstance(e.scalar_compute, ArgReduceCompute):
                self.visit(e.scalar_compute.value)
            else:
                raise NotImplementedError()

    def visit_TensorNode(self, e: TensorNode):
        if e.tensor_compute:
            if isinstance(e.tensor_compute, GridCompute):
                self.visit(e.tensor_compute.value)
            else:
                raise NotImplementedError()

    # low level dialect
    def visit_Cast(self, e: Cast):
        self.visit(e.expr)

    def visit_Dereference(self, e: Dereference):
        self.visit(e.expr)

    def visit_Address(self, e: Address):
        self.visit(e.expr)

    def visit_Reference(self, e: Reference):
        self.visit(e.expr)

    def visit_AnyExpr(self, e: AnyExpr):
        pass


class ExprRewriter(ExprFunctor):
    def rewrite(self, e):
        return self.visit(e)

    def visit_Binary(self, e: BinaryOp):
        a = self(e.a)
        b = self(e.b)
        if a is e.a and b is e.b:
            return e
        else:
            return e.__class__(a, b)

    def visit_Add(self, e: Add):
        return self.visit_Binary(e)

    def visit_Sub(self, e: Sub):
        return self.visit_Binary(e)

    def visit_Multiply(self, e: Multiply):
        return self.visit_Binary(e)

    def visit_Div(self, e: Div):
        return self.visit_Binary(e)

    def visit_Mod(self, e: Mod):
        return self.visit_Binary(e)

    def visit_FloorDiv(self, e: FloorDiv):
        return self.visit_Binary(e)

    def visit_LessThan(self, e: LessThan):
        return self.visit_Binary(e)

    def visit_LessEqual(self, e: LessEqual):
        return self.visit_Binary(e)

    def visit_Equal(self, e: Equal):
        return self.visit_Binary(e)

    def visit_NotEqual(self, e: NotEqual):
        return self.visit_Binary(e)

    def visit_And(self, e: LogicalAnd):
        return self.visit_Binary(e)

    def visit_Or(self, e: LogicalOr):
        return self.visit_Binary(e)

    def visit_Neg(self, e: Neg):
        a = self(e.a)
        if a is e.a:
            return e
        else:
            return Neg(a)

    def visit_Not(self, e: LogicalNot):
        a = self(e.a)
        if a is e.a:
            return e
        else:
            return LogicalNot(a)

    def visit_BitwiseAnd(self, e: BitwiseAnd):
        return self.visit_Binary(e)

    def visit_BitwiseOr(self, e: BitwiseOr):
        return self.visit_Binary(e)

    def visit_BitwiseXor(self, e: BitwiseXor):
        return self.visit_Binary(e)

    def visit_BitwiseNot(self, e: BitwiseNot):
        base = self.visit(e.base)
        if base is e.base:
            return e
        else:
            return BitwiseNot(base)

    def visit_LeftShift(self, e: LeftShift):
        base = self.visit(e.base)
        cnt = self.visit(e.cnt)
        if base is e.base and cnt is e.cnt:
            return e
        else:
            return LeftShift(base, cnt)

    def visit_RightShift(self, e: RightShift):
        base = self.visit(e.base)
        cnt = self.visit(e.cnt)
        if base is e.base and cnt is e.cnt:
            return e
        else:
            return RightShift(base, cnt)

    def visit_TensorElement(self, e: TensorElement):
        base = self(e.base)
        indices = [self(idx) if idx is not None else None for idx in e.indices]
        if base is e.base and same_list(indices, e.indices):
            return e
        else:
            return TensorElement(base, indices, e.protected)

    def visit_TensorSlice(self, e: TensorSlice):
        base = self(e.base)
        indices = [self(idx) if idx is not None else None for idx in e.indices]
        starts = [self(start) if start is not None else None for start in e.starts]
        ends = [self(end) if end is not None else None for end in e.ends]
        if base is e.base and same_list(indices, e.indices) and same_list(starts, e.starts) and same_list(ends, e.ends):
            return e
        else:
            return TensorSlice(base, indices, starts, ends)

    def visit_IfThenElse(self, e: IfThenElse):
        cond = self(e.cond)
        then_expr = self(e.then_expr)
        else_expr = self(e.else_expr)
        if cond is e.cond and then_expr is e.then_expr and else_expr is e.else_expr:
            return e
        else:
            return IfThenElse(cond, then_expr, else_expr)

    def visit_Cast(self, e: Cast):
        expr = self(e.expr)
        if expr is e.expr:
            return e
        else:
            return Cast(expr, e.target_type)

    def visit_Dereference(self, e: Dereference):
        expr = self(e.expr)
        if expr is e.expr:
            return e
        else:
            return Dereference(expr)

    def visit_Address(self, e: Address):
        expr = self(e.expr)
        if expr is e.expr:
            return e
        else:
            return Address(expr)

    def visit_Reference(self, e: Reference):
        expr = self(e.expr)
        if expr is e.expr:
            return e
        else:
            return Reference(expr)

    def visit_Call(self, e: Call):
        func_var = self(e.func_var)
        args = [self(arg) for arg in e.args]
        if func_var is e.func_var and same_list(args, e.args):
            return e
        else:
            return Call(func_var, args)

    def visit_Let(self, e: Let):
        v = e.var
        value = self(e.value)
        body = self(e.body)
        if same_list([v, value, body], [e.var, e.value, e.body]):
            return e
        else:
            return Let(v, value, body)

    def visit_Var(self, e: Var):
        return e

    def visit_Constant(self, e: Constant):
        return e

    def visit_ScalarNode(self, e: ScalarNode):
        from hidet.ir.functors import collect  # pylint: disable=import-outside-toplevel

        if e.scalar_compute is None:
            return e
        else:
            if isinstance(e.scalar_compute, ReduceCompute):
                rc = e.scalar_compute
                axes = self(rc.axes)
                value = self(rc.value)
                shape = self(rc.shape)
                if value is rc.value and same_list(axes, rc.axes) and same_list(shape, rc.shape):
                    return e
                else:
                    input_tensors = collect(value, TensorNode, stop_when_found=True)
                    input_scalars = collect(value, ScalarNode, stop_when_found=True)
                    return ScalarNode(
                        e.name,
                        e.dtype,
                        ReduceCompute(
                            input_tensors, input_scalars, shape, axes, value, rc.reduce_operation, rc.accumulate_dtype
                        ),
                    )
            elif isinstance(e.scalar_compute, ArgReduceCompute):
                sc = e.scalar_compute
                axis = self(sc.axis)
                value = self(sc.value)
                extent = self(sc.extent)
                if value is sc.value and axis is sc.axis and extent is sc.extent:
                    return e
                else:
                    input_tensors = collect(value, TensorNode, stop_when_found=True)
                    input_scalars = collect(value, ScalarNode, stop_when_found=True)
                    return ScalarNode(
                        e.name,
                        e.dtype,
                        ArgReduceCompute(
                            input_tensors, input_scalars, extent, axis, value, sc.reduce_operation, sc.index_dtype
                        ),
                    )
            else:
                raise NotImplementedError('Can not recognize ScalarCompute: {}'.format(type(e.scalar_compute).__name__))

    def visit_TensorNode(self, e: TensorNode):
        from hidet.ir.functors import collect  # pylint: disable=import-outside-toplevel

        if e.tensor_compute is None:
            return e
        else:
            if isinstance(e.tensor_compute, GridCompute):
                gc = e.tensor_compute
                axes = self(gc.axes)
                value = self(gc.value)
                shape = self(gc.shape)
                if value is gc.value and same_list(axes, gc.axes) and same_list(shape, gc.shape):
                    return e
                else:
                    input_tensors = collect(value, TensorNode, stop_when_found=True)
                    input_scalars = collect(value, ScalarNode, stop_when_found=True)
                    return TensorNode(e.name, e.ttype, GridCompute(input_tensors, input_scalars, shape, axes, value))
            else:
                raise NotImplementedError('Can not recognize TensorCompute: {}'.format(type(e.tensor_compute).__name__))

    def visit_AnyExpr(self, e: AnyExpr):
        return e


class StmtFunctor(NodeFunctor):
    @staticmethod
    def get_dispatch_mapping(cls) -> Mapping[Type[Node], Any]:
        return {
            EvaluateStmt: cls.visit_EvaluateStmt,
            DeclareStmt: cls.visit_DeclareStmt,
            BufferStoreStmt: cls.visit_BufferStoreStmt,
            AssignStmt: cls.visit_AssignStmt,
            LetStmt: cls.visit_LetStmt,
            ForStmt: cls.visit_ForStmt,
            ForTaskStmt: cls.visit_ForTaskStmt,
            WhileStmt: cls.visit_WhileStmt,
            BreakStmt: cls.visit_BreakStmt,
            ContinueStmt: cls.visit_ContinueStmt,
            IfStmt: cls.visit_IfStmt,
            ReturnStmt: cls.visit_ReturnStmt,
            AsmStmt: cls.visit_AsmStmt,
            LaunchKernelStmt: cls.visit_LaunchKernelStmt,
            AssertStmt: cls.visit_AssertStmt,
            BlackBoxStmt: cls.visit_BlackBoxStmt,
            SeqStmt: cls.visit_SeqStmt,
        }

    def visit_expr(self, e: Expr):
        raise NotImplementedError()

    def visit_DeclareStmt(self, stmt: DeclareStmt):
        raise NotImplementedError()

    def visit_EvaluateStmt(self, stmt: EvaluateStmt):
        raise NotImplementedError()

    def visit_BufferStoreStmt(self, stmt: BufferStoreStmt):
        raise NotImplementedError()

    def visit_AssignStmt(self, stmt: AssignStmt):
        raise NotImplementedError()

    def visit_LetStmt(self, stmt: LetStmt):
        raise NotImplementedError()

    def visit_ForStmt(self, stmt: ForStmt):
        raise NotImplementedError()

    def visit_ForTaskStmt(self, stmt: ForTaskStmt):
        raise NotImplementedError()

    def visit_WhileStmt(self, stmt: WhileStmt):
        raise NotImplementedError()

    def visit_BreakStmt(self, stmt: BreakStmt):
        raise NotImplementedError()

    def visit_ContinueStmt(self, stmt: ContinueStmt):
        raise NotImplementedError()

    def visit_IfStmt(self, stmt: IfStmt):
        raise NotImplementedError()

    def visit_ReturnStmt(self, stmt: ReturnStmt):
        raise NotImplementedError()

    def visit_AssertStmt(self, stmt: AssertStmt):
        raise NotImplementedError()

    def visit_AsmStmt(self, stmt: AsmStmt):
        raise NotImplementedError()

    def visit_LaunchKernelStmt(self, stmt: LaunchKernelStmt):
        raise NotImplementedError()

    def visit_BlackBoxStmt(self, stmt: BlackBoxStmt):
        raise NotImplementedError()

    def visit_SeqStmt(self, stmt: SeqStmt):
        raise NotImplementedError()


class StmtVisitor(StmtFunctor):
    def visit_expr(self, e: Expr):
        pass

    def visit_DeclareStmt(self, stmt: DeclareStmt):
        self.visit_expr(stmt.var)
        if stmt.init:
            self.visit_expr(stmt.init)

    def visit_EvaluateStmt(self, stmt: EvaluateStmt):
        self.visit_expr(stmt.expr)

    def visit_BufferStoreStmt(self, stmt: BufferStoreStmt):
        self.visit_expr(stmt.buf)
        self.visit_expr(stmt.value)
        for idx in stmt.indices:
            self.visit_expr(idx)

    def visit_AssignStmt(self, stmt: AssignStmt):
        self.visit_expr(stmt.var)
        self.visit_expr(stmt.value)

    def visit_LetStmt(self, stmt: LetStmt):
        for _, bind_value in zip(stmt.bind_vars, stmt.bind_values):
            self.visit_expr(bind_value)
        self.visit(stmt.body)

    def visit_ForStmt(self, stmt: ForStmt):
        self.visit_expr(stmt.extent)
        self.visit(stmt.body)

    def visit_ForTaskStmt(self, stmt: ForTaskStmt):
        for loop_var in stmt.loop_vars:
            self.visit_expr(loop_var)
        self.visit_expr(stmt.worker)
        self.visit(stmt.body)

    def visit_WhileStmt(self, stmt: WhileStmt):
        self.visit(stmt.cond)
        self.visit(stmt.body)

    def visit_BreakStmt(self, stmt: BreakStmt):
        pass

    def visit_ContinueStmt(self, stmt: ContinueStmt):
        pass

    def visit_IfStmt(self, stmt: IfStmt):
        self.visit_expr(stmt.cond)
        self.visit(stmt.then_body)
        if stmt.else_body:
            self.visit(stmt.else_body)

    def visit_ReturnStmt(self, stmt: ReturnStmt):
        self.visit(stmt.ret_value)

    def visit_AssertStmt(self, stmt: AssertStmt):
        self.visit(stmt.cond)

    def visit_AsmStmt(self, stmt: AsmStmt):
        for expr in stmt.input_exprs:
            self.visit_expr(expr)
        for expr in stmt.output_exprs:
            self.visit_expr(expr)

    def visit_LaunchKernelStmt(self, stmt: LaunchKernelStmt):
        self.visit_expr(stmt.func_var)
        for arg in stmt.args:
            self.visit_expr(arg)
        for dim in stmt.grid_dim:
            self.visit_expr(dim)
        for dim in stmt.block_dim:
            self.visit_expr(dim)
        self.visit_expr(stmt.shared_mem_bytes)

    def visit_BlackBoxStmt(self, stmt: BlackBoxStmt):
        for expr in stmt.exprs:
            self.visit_expr(expr)

    def visit_SeqStmt(self, stmt: SeqStmt):
        for s in stmt.seq:
            self.visit(s)


class StmtRewriter(StmtFunctor):
    def visit_expr(self, e: Expr):
        return e

    def visit_DeclareStmt(self, stmt: DeclareStmt):
        v = self.visit_expr(stmt.var)
        init = self.visit_expr(stmt.init) if stmt.init else None
        if v is stmt.var and init is stmt.init:
            return stmt
        else:
            return DeclareStmt(v, init, stmt.is_static)

    def visit_EvaluateStmt(self, stmt: EvaluateStmt):
        e = self.visit_expr(stmt.expr)
        if e is stmt.expr:
            return stmt
        else:
            return EvaluateStmt(e)

    def visit_BufferStoreStmt(self, stmt: BufferStoreStmt):
        buf = self.visit_expr(stmt.buf)
        indices = [self.visit_expr(e) for e in stmt.indices]
        value = self.visit_expr(stmt.value)
        if buf is stmt.buf and all(a is b for a, b in zip(indices, stmt.indices)) and value is stmt.value:
            return stmt
        else:
            return BufferStoreStmt(buf, indices, value, stmt.protected)

    def visit_AssignStmt(self, stmt: AssignStmt):
        v = self.visit_expr(stmt.var)
        value = self.visit_expr(stmt.value)
        if v is stmt.var and value is stmt.value:
            return stmt
        else:
            return AssignStmt(v, value)

    def visit_LetStmt(self, stmt: LetStmt):
        bind_values = [self.visit_expr(bind_value) for bind_value in stmt.bind_values]
        body = self.visit(stmt.body)
        if same_list(bind_values, stmt.bind_values) and body is stmt.body:
            return stmt
        else:
            return LetStmt(stmt.bind_vars, bind_values, body)

    def visit_ForStmt(self, stmt: ForStmt):
        loop_var = stmt.loop_var
        extent = self.visit_expr(stmt.extent)
        body = self.visit(stmt.body)
        if loop_var is stmt.loop_var and extent is stmt.extent and body is stmt.body:
            return stmt
        else:
            return ForStmt(loop_var, extent, stmt.unroll, body)

    def visit_ForTaskStmt(self, stmt: ForTaskStmt):
        loop_vars: List[Expr] = [self.visit_expr(v) for v in stmt.loop_vars]
        # todo: visit expressions in task mapping
        worker = self.visit_expr(stmt.worker)
        body = self.visit(stmt.body)
        if same_list(loop_vars, stmt.loop_vars) and worker is stmt.worker and body is stmt.body:
            return stmt
        else:
            assert all(isinstance(v, Var) for v in loop_vars)
            asserted_loop_vars: List[Var] = [v for v in loop_vars if isinstance(v, Var)]  # avoid IDE warning
            return ForTaskStmt(loop_vars=asserted_loop_vars, mapping=stmt.mapping, worker=worker, body=body)

    def visit_WhileStmt(self, stmt: WhileStmt):
        cond = self.visit(stmt.cond)
        body = self.visit(stmt.body)
        if cond is stmt.cond and body is stmt.body:
            return stmt
        else:
            return WhileStmt(cond, body)

    def visit_BreakStmt(self, stmt: BreakStmt):
        return stmt

    def visit_ContinueStmt(self, stmt: ContinueStmt):
        return stmt

    def visit_IfStmt(self, stmt: IfStmt):
        cond = self.visit_expr(stmt.cond)
        then_body = self.visit(stmt.then_body)
        else_body = self.visit(stmt.else_body) if stmt.else_body else None
        if cond is stmt.cond and then_body is stmt.then_body and else_body is stmt.else_body:
            return stmt
        else:
            return IfStmt(cond, then_body, else_body)

    def visit_ReturnStmt(self, stmt: ReturnStmt):
        ret_value = self.visit_expr(stmt.ret_value) if stmt.ret_value is not None else None
        if ret_value is stmt.ret_value:
            return stmt
        else:
            return ReturnStmt(ret_value)

    def visit_AssertStmt(self, stmt: AssertStmt):
        cond = self.visit_expr(stmt.cond)
        if cond is stmt.cond:
            return stmt
        else:
            return AssertStmt(cond, stmt.msg)

    def visit_AsmStmt(self, stmt: AsmStmt):
        input_exprs = [self.visit_expr(e) for e in stmt.input_exprs]
        output_exprs = [self.visit_expr(e) for e in stmt.output_exprs]
        if same_list(input_exprs, stmt.input_exprs) and same_list(output_exprs, stmt.output_exprs):
            return stmt
        else:
            return AsmStmt(
                stmt.template_string,
                list(zip(stmt.output_labels, output_exprs)),
                list(zip(stmt.input_labels, input_exprs)),
                stmt.is_volatile,
            )

    def visit_LaunchKernelStmt(self, stmt: LaunchKernelStmt):
        func_var = self.visit_expr(stmt.func_var)
        args = [self.visit_expr(e) for e in stmt.args]
        grid_dim = (
            self.visit_expr(stmt.grid_dim[0]),
            self.visit_expr(stmt.grid_dim[1]),
            self.visit_expr(stmt.grid_dim[2]),
        )
        block_dim = (
            self.visit_expr(stmt.block_dim[0]),
            self.visit_expr(stmt.block_dim[1]),
            self.visit_expr(stmt.block_dim[2]),
        )
        shared_mem_bytes = self.visit_expr(stmt.shared_mem_bytes)
        if same_list(
            [func_var, *args, *grid_dim, *block_dim, shared_mem_bytes],
            [stmt.func_var, *stmt.args, *stmt.grid_dim, *stmt.block_dim, stmt.shared_mem_bytes],
        ):
            return stmt
        else:
            return LaunchKernelStmt(func_var, args, grid_dim, block_dim, shared_mem_bytes)

    def visit_BlackBoxStmt(self, stmt: BlackBoxStmt):
        exprs = [self.visit_expr(e) for e in stmt.exprs]
        if same_list(exprs, stmt.exprs):
            return stmt
        else:
            return BlackBoxStmt(stmt.template_string, *exprs)

    def visit_SeqStmt(self, stmt: SeqStmt):
        seq = []
        for s in stmt.seq:
            seq.append(self.visit(s))
        if all(a is b for a, b in zip(seq, stmt.seq)):
            return stmt
        else:
            return SeqStmt(seq)


class StmtExprFunctor(ExprFunctor, StmtFunctor):  # pylint: disable=abstract-method
    def visit_expr(self, e: Expr):
        return self.visit(e)


class StmtExprVisitor(ExprVisitor, StmtVisitor):
    def visit_expr(self, e: Expr):
        return self.visit(e)


class FuncStmtExprVisitor(StmtExprVisitor):
    @staticmethod
    def get_dispatch_mapping(cls) -> Mapping[Type[Node], Any]:
        return {Function: cls.visit_Function}

    def visit_Function(self, func: Function):
        self(func.body)


class StmtExprRewriter(ExprRewriter, StmtRewriter):
    def visit_expr(self, e: Expr):
        return self.visit(e)


class FuncStmtExprRewriter(StmtExprRewriter):
    @staticmethod
    def get_dispatch_mapping(cls) -> Mapping[Type[Node], Any]:
        return {Function: cls.visit_Function}

    def visit_Function(self, func: Function):
        body = self(func.body)
        if body is func.body:
            return func
        else:
            return Function(
                func.name,
                params=func.params,
                body=body,
                ret_type=func.ret_type,
                kind=func.kind,
                extern_vars=func.extern_vars,
                attrs=func.attrs,
            )


class TypeFunctor(NodeFunctor):
    @staticmethod
    def get_dispatch_mapping(cls) -> Mapping[Type[Node], Any]:
        return {
            DataType: cls.visit_ScalarType,
            TensorType: cls.visit_TensorType,
            PointerType: cls.visit_PointerType,
            TensorPointerType: cls.visit_TensorPointerType,
            ReferenceType: cls.visit_ReferenceType,
            VoidType: cls.visit_VoidType,
        }

    def visit_ScalarType(self, t: DataType):
        raise NotImplementedError()

    def visit_TensorType(self, t: TensorType):
        raise NotImplementedError()

    def visit_PointerType(self, t: PointerType):
        raise NotImplementedError()

    def visit_TensorPointerType(self, t: TensorPointerType):
        raise NotImplementedError()

    def visit_ReferenceType(self, t: ReferenceType):
        raise NotImplementedError()

    def visit_VoidType(self, t: VoidType):
        raise NotImplementedError()


class ComputeFunctor(NodeFunctor):
    @staticmethod
    def get_dispatch_mapping(cls) -> Mapping[Type[Node], Any]:
        return {
            ScalarNode: cls.visit_ScalarNode,
            TensorNode: cls.visit_TensorNode,
            GridCompute: cls.visit_GridCompute,
            ReduceCompute: cls.visit_ReduceCompute,
            ArgReduceCompute: cls.visit_ArgReduceCompute,
        }

    def visit_ScalarNode(self, node: ScalarNode):
        raise NotImplementedError()

    def visit_TensorNode(self, node: TensorNode):
        raise NotImplementedError()

    def visit_GridCompute(self, c: GridCompute):
        raise NotImplementedError()

    def visit_ReduceCompute(self, c: ReduceCompute):
        raise NotImplementedError()

    def visit_ArgReduceCompute(self, c: ArgReduceCompute):
        raise NotImplementedError()


def same_list(lhs: Sequence, rhs: Sequence):
    if len(lhs) != len(rhs):
        return False
    return all(a is b for a, b in zip(lhs, rhs))
