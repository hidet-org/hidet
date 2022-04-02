from abc import ABC
from typing import Mapping

from hidet.ir.dialects.pattern import *
from hidet.ir.func import *
from hidet.ir.stmt import *


class NodeFunctor:
    def __init__(self, use_memo=True):
        self.memo = {} if use_memo else None
        if not hasattr(self.__class__, 'dispatch_table'):
            self.setup_dispatch_table()

    def __call__(self, node: Any):
        return self.visit(node)

    def visit(self, node: Node):
        if self.memo is not None and node in self.memo:
            return self.memo[node]
        if isinstance(node, Node):
            idx = node.class_index() if node is not None else 0
            # noinspection PyUnresolvedReferences
            dispatch_table = self.__class__.dispatch_table
            if idx >= len(dispatch_table):
                raise NotImplementedError('Does not implement dispatch function in "{}" for node "{}"'.format(type(self).__qualname__, type(node).__qualname__))
            ret = dispatch_table[idx](self, node)
        elif isinstance(node, tuple):
            ret = tuple(self.visit(v) for v in node)
        else:
            raise NotImplementedError()
        if self.memo is not None:
            self.memo[node] = ret
        return ret

    @staticmethod
    def get_dispatch_mapping(cls) -> Mapping[Type[Node], Any]:
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
            And: cls.visit_And,
            Or: cls.visit_Or,
            Not: cls.visit_Not,
            BitwiseAnd: cls.visit_BitwiseAnd,
            BitwiseOr: cls.visit_BitwiseOr,
            BitwiseNot: cls.visit_BitwiseNot,
            LeftShift: cls.visit_LeftShift,
            RightShift: cls.visit_RightShift,
            TensorElement: cls.visit_TensorElement,
            TensorSlice: cls.visit_TensorSlice,
            IfThenElse: cls.visit_IfThenElse,
            AlterLayout: cls.visit_AlterLayout,
            Call: cls.visit_Call,
            Let: cls.visit_Let,
            Var: cls.visit_Var,
            Constant: cls.visit_Constant,
            Cast: cls.visit_Cast,
            Dereference: cls.visit_Dereference,
            Address: cls.visit_Address,
            Reference: cls.visit_Reference,
            ScalarInput: cls.visit_ScalarInput,
            TensorInput: cls.visit_TensorInput,
            TensorCompute: cls.visit_TensorCompute,
            ReduceCompute: cls.visit_ReduceCompute,
            AnyExpr: cls.visit_AnyExpr,
            ReduceComputePattern: cls.visit_ReduceComputePattern,
            TensorComputePattern: cls.visit_TensorComputePattern,
            ScalarExprPattern: cls.visit_ScalarExprPattern,
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

    def visit_And(self, e: And):
        raise NotImplementedError()

    def visit_Or(self, e: Or):
        raise NotImplementedError()

    def visit_Neg(self, e: Neg):
        raise NotImplementedError()

    def visit_Not(self, e: Not):
        raise NotImplementedError()

    def visit_BitwiseAnd(self, e: BitwiseAnd):
        raise NotImplementedError()

    def visit_BitwiseOr(self, e: BitwiseOr):
        raise NotImplementedError()

    def visit_BitwiseNot(self, e: BitwiseNot):
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

    def visit_AlterLayout(self, e: AlterLayout):
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

    def visit_ScalarInput(self, e: ScalarInput):
        raise NotImplementedError()

    def visit_TensorInput(self, e: TensorInput):
        raise NotImplementedError()

    def visit_TensorCompute(self, e: TensorCompute):
        raise NotImplementedError()

    def visit_ReduceCompute(self, e: ReduceCompute):
        raise NotImplementedError()

    def visit_AnyExpr(self, e: AnyExpr):
        raise NotImplementedError()

    def visit_ReduceComputePattern(self, e: ReduceComputePattern):
        raise NotImplementedError()

    def visit_TensorComputePattern(self, e: TensorComputePattern):
        raise NotImplementedError()

    def visit_ScalarExprPattern(self, e: ScalarExprPattern):
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

    def visit_LessEqual(self, e: LessThan):
        self.visit(e.a)
        self.visit(e.b)

    def visit_Equal(self, e: Equal):
        self.visit(e.a)
        self.visit(e.b)

    def visit_And(self, e: And):
        self.visit(e.a)
        self.visit(e.b)

    def visit_Or(self, e: Or):
        self.visit(e.a)
        self.visit(e.b)

    def visit_Neg(self, e: Neg):
        self.visit(e.a)

    def visit_Not(self, e: Not):
        self.visit(e.a)

    def visit_BitwiseAnd(self, e: BitwiseAnd):
        self.visit(e.a)
        self.visit(e.b)

    def visit_BitwiseOr(self, e: BitwiseOr):
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

    def visit_AlterLayout(self, e: AlterLayout):
        self.visit(e.var)

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
    def visit_ScalarInput(self, e: ScalarInput):
        pass

    def visit_TensorInput(self, e: TensorInput):
        pass

    def visit_TensorCompute(self, e: TensorCompute):
        self.visit(e.value)

    def visit_ReduceCompute(self, e: ReduceCompute):
        self.visit(e.value)

    # pattern dialect
    def visit_AnyExpr(self, e: ReduceComputePattern):
        pass

    def visit_ReduceComputePattern(self, e: ReduceComputePattern):
        pass

    def visit_TensorComputePattern(self, e: TensorComputePattern):
        pass

    def visit_ScalarExprPattern(self, e: ScalarExprPattern):
        pass

    # lowlevel dialect
    def visit_Cast(self, e: Cast):
        self.visit(e.expr)

    def visit_Dereference(self, e: Dereference):
        self.visit(e.expr)

    def visit_Address(self, e: Address):
        self.visit(e.expr)

    def visit_Reference(self, e: Reference):
        self.visit(e.expr)


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

    def visit_And(self, e: And):
        return self.visit_Binary(e)

    def visit_Or(self, e: Or):
        return self.visit_Binary(e)

    def visit_Neg(self, e: Neg):
        a = self(e.a)
        if a is e.a:
            return e
        else:
            return Neg(a)

    def visit_Not(self, e: Not):
        a = self(e.a)
        if a is e.a:
            return e
        else:
            return Not(a)

    def visit_BitwiseAnd(self, e: BitwiseAnd):
        return self.visit_Binary(e)

    def visit_BitwiseOr(self, e: BitwiseOr):
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
            return TensorElement(base, indices)

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

    def visit_AlterLayout(self, e: AlterLayout):
        var = self(e.var)
        if var is e.var:
            return e
        else:
            return AlterLayout(var, e.shape, e.layout_map)

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
        var = e.var
        value = self(e.value)
        body = self(e.body)
        if same_list([var, value, body], [e.var, e.value, e.body]):
            return e
        else:
            return Let(var, value, body)

    def visit_Var(self, e: Var):
        return e

    def visit_Constant(self, e: Constant):
        return e

    def visit_ScalarInput(self, e: ScalarInput):
        return e

    def visit_TensorInput(self, e: TensorInput):
        return e

    def visit_TensorCompute(self, e: TensorCompute):
        name = e.name
        value = self(e.value)
        axes = [self(axis) for axis in e.axes]
        shape = [self(s) for s in e.shape]
        if value is e.value and same_list(axes, e.axes) and same_list(shape, e.shape):
            return e
        else:
            return TensorCompute(name, shape, axes, value)

    def visit_ReduceCompute(self, e: ReduceCompute):
        value = self(e.value)
        axes = [self(axis) for axis in e.axes]
        shape = [self(v) for v in e.shape]
        if value is e.value and same_list(axes, e.axes) and same_list(shape, e.shape):
            return e
        else:
            return ReduceCompute(value, shape, axes, e.reduce_type)

    def visit_AnyExpr(self, e: AnyExpr):
        return e

    def visit_ReduceComputePattern(self, e: ReduceComputePattern):
        return e

    def visit_TensorComputePattern(self, e: TensorComputePattern):
        return e

    def visit_ScalarExprPattern(self, e: ScalarExprPattern):
        return e


class StmtFunctor(NodeFunctor):
    @staticmethod
    def get_dispatch_mapping(cls) -> Mapping[Type[Node], Any]:
        return {
            EvaluateStmt: cls.visit_EvaluateStmt,
            BufferStoreStmt: cls.visit_BufferStoreStmt,
            AssignStmt: cls.visit_AssignStmt,
            LetStmt: cls.visit_LetStmt,
            ForStmt: cls.visit_ForStmt,
            IfStmt: cls.visit_IfStmt,
            ReturnStmt: cls.visit_ReturnStmt,
            AsmStmt: cls.visit_AsmStmt,
            AssertStmt: cls.visit_AssertStmt,
            BlackBoxStmt: cls.visit_BlackBoxStmt,
            SeqStmt: cls.visit_SeqStmt,
        }

    def visit_expr(self, e: Expr):
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

    def visit_IfStmt(self, stmt: IfStmt):
        raise NotImplementedError()

    def visit_ReturnStmt(self, stmt: ReturnStmt):
        raise NotImplementedError()

    def visit_AssertStmt(self, stmt: AssertStmt):
        raise NotImplementedError()

    def visit_AsmStmt(self, stmt: AsmStmt):
        raise NotImplementedError()

    def visit_BlackBoxStmt(self, stmt: BlackBoxStmt):
        raise NotImplementedError()

    def visit_SeqStmt(self, stmt: SeqStmt):
        raise NotImplementedError()


class StmtVisitor(StmtFunctor):
    def visit_expr(self, e: Expr):
        pass

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
        for bind_var, bind_value in zip(stmt.bind_vars, stmt.bind_values):
            self.visit_expr(bind_value)
        self.visit(stmt.body)

    def visit_ForStmt(self, stmt: ForStmt):
        self.visit_expr(stmt.extent)
        self.visit(stmt.body)

    def visit_IfStmt(self, stmt: IfStmt):
        self.visit_expr(stmt.cond)
        self.visit(stmt.then_body)
        if stmt.else_body:
            self.visit(stmt.else_body)

    def visit_ReturnStmt(self, stmt: ReturnStmt):
        pass

    def visit_AssertStmt(self, stmt: AssertStmt):
        self.visit(stmt.cond)

    def visit_AsmStmt(self, stmt: AsmStmt):
        for expr in stmt.input_exprs:
            self.visit_expr(expr)
        for expr in stmt.output_exprs:
            self.visit_expr(expr)

    def visit_BlackBoxStmt(self, stmt: BlackBoxStmt):
        for expr in stmt.exprs:
            self.visit_expr(expr)

    def visit_SeqStmt(self, stmt: SeqStmt):
        for s in stmt.seq:
            self.visit(s)


class StmtRewriter(StmtFunctor):
    def visit_expr(self, e: Expr):
        return e

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
            return BufferStoreStmt(buf, indices, value)

    def visit_AssignStmt(self, stmt: AssignStmt):
        var = self.visit_expr(stmt.var)
        value = self.visit_expr(stmt.value)
        if var is stmt.var and value is stmt.value:
            return stmt
        else:
            return AssignStmt(var, value)

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
        if loop_var is stmt.loop_var and body is stmt.body:
            return stmt
        else:
            return ForStmt(loop_var, extent, stmt.unroll, body)

    def visit_IfStmt(self, stmt: IfStmt):
        cond = self.visit_expr(stmt.cond)
        then_body = self.visit(stmt.then_body)
        else_body = self.visit(stmt.else_body) if stmt.else_body else None
        if cond is stmt.cond and then_body is stmt.then_body and else_body is stmt.else_body:
            return stmt
        else:
            return IfStmt(cond, then_body, else_body)

    def visit_ReturnStmt(self, stmt: ReturnStmt):
        return stmt

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
            return AsmStmt(stmt.template_string, list(zip(stmt.output_labels, output_exprs)),
                           list(zip(stmt.input_labels, input_exprs)), stmt.is_volatile)

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


class StmtExprFunctor(ExprFunctor, StmtFunctor):
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
            return Function(func.name, func.params, body, func.ret_type, func.local_vars, func.extern_vars, func.attrs)


class TypeFunctor(NodeFunctor):
    @staticmethod
    def get_dispatch_mapping(cls) -> Mapping[Type[Node], Any]:
        return {
            ScalarType: cls.visit_ScalarType,
            TensorType: cls.visit_TensorType,
            PointerType: cls.visit_PointerType,
            TensorPointerType: cls.visit_TensorPointerType,
            ReferenceType: cls.visit_ReferenceType,
            VoidType: cls.visit_VoidType,
        }

    def visit_ScalarType(self, t: ScalarType):
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


class WorkerFunctor(NodeFunctor):
    @staticmethod
    def get_dispatch_mapping(cls) -> Mapping[Type[Node], Any]:
        return {
            Host: cls.visit_Host,
            Grid: cls.visit_Grid,
            ThreadBlock: cls.visit_ThreadBlock,
            Warp: cls.visit_Warp,
            Thread: cls.visit_Thread,
        }

    def visit_Host(self, host: Host):
        raise NotImplementedError()

    def visit_Grid(self, grid: Grid):
        raise NotImplementedError()

    def visit_ThreadBlock(self, block: ThreadBlock):
        raise NotImplementedError()

    def visit_Warp(self, warp: Warp):
        raise NotImplementedError()

    def visit_Thread(self, thread: Thread):
        raise NotImplementedError()


def same_list(lhs: Sequence, rhs: Sequence):
    if len(lhs) != len(rhs):
        return False
    return all(a is b for a, b in zip(lhs, rhs))
