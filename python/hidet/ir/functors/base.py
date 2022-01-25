from hidet.ir.type import *
from hidet.ir.expr import *
from hidet.ir.stmt import *
from hidet.ir.task import *
from hidet.ir.dialects.compute import *
from hidet.ir.dialects.lowlevel import *
from hidet.ir.dialects.pattern import *


class ExprFunctor:
    def __init__(self):
        self.memo = {}

    def __call__(self, *args, **kwargs):
        return self.visit(*args, **kwargs)

    def visit(self, e):
        if e in self.memo:
            return self.memo[e]
        if isinstance(e, Add):
            res = self.visit_Add(e)
        elif isinstance(e, Sub):
            res = self.visit_Sub(e)
        elif isinstance(e, Multiply):
            res = self.visit_Multiply(e)
        elif isinstance(e, Div):
            res = self.visit_Div(e)
        elif isinstance(e, Mod):
            res = self.visit_Mod(e)
        elif isinstance(e, FloorDiv):
            res = self.visit_FloorDiv(e)
        elif isinstance(e, LessThan):
            res = self.visit_LessThan(e)
        elif isinstance(e, LessEqual):
            res = self.visit_LessEqual(e)
        elif isinstance(e, Equal):
            res = self.visit_Equal(e)
        elif isinstance(e, And):
            res = self.visit_And(e)
        elif isinstance(e, Or):
            res = self.visit_Or(e)
        elif isinstance(e, Not):
            res = self.visit_Not(e)
        elif isinstance(e, TensorSlice):
            res = self.visit_TensorSlice(e)
        elif isinstance(e, TensorElement):
            res = self.visit_TensorElement(e)
        elif isinstance(e, Call):
            res = self.visit_Call(e)
        elif isinstance(e, Var):
            res = self.visit_Var(e)
        elif isinstance(e, Constant):
            res = self.visit_Constant(e)
        # lowlevel dialect
        elif isinstance(e, Cast):
            res = self.visit_Cast(e)
        elif isinstance(e, Dereference):
            res = self.visit_Dereference(e)
        elif isinstance(e, Address):
            res = self.visit_Address(e)
        elif isinstance(e, Reference):
            res = self.visit_Reference(e)
        # compute dialect
        elif isinstance(e, ScalarInput):
            res = self.visit_ScalarInput(e)
        elif isinstance(e, TensorInput):
            res = self.visit_TensorInput(e)
        elif isinstance(e, TensorCompute):
            res = self.visit_TensorCompute(e)
        elif isinstance(e, ReduceCompute):
            res = self.visit_ReduceCompute(e)
        # pattern dialect
        elif isinstance(e, AnyExpr):
            res = self.visit_AnyExpr(e)
        elif isinstance(e, ReduceComputePattern):
            res = self.visit_ReduceComputePattern(e)
        elif isinstance(e, TensorComputePattern):
            res = self.visit_TensorComputePattern(e)
        elif isinstance(e, ScalarExprPattern):
            res = self.visit_ScalarExprPattern(e)
        else:
            raise NotImplementedError()
        self.memo[e] = res
        return res

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

    def visit_LessEqual(self, e: LessThan):
        raise NotImplementedError()

    def visit_Equal(self, e: Equal):
        raise NotImplementedError()

    def visit_And(self, e: And):
        raise NotImplementedError()

    def visit_Or(self, e: Or):
        raise NotImplementedError()

    def visit_Not(self, e: Not):
        raise NotImplementedError()

    def visit_TensorSlice(self, e: TensorSlice):
        raise NotImplementedError()

    def visit_TensorElement(self, e: TensorElement):
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

    def visit_AnyExpr(self, e: ReduceComputePattern):
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

    def visit_Not(self, e: Not):
        self.visit(e.a)

    def visit_TensorSlice(self, e: TensorSlice):
        self.visit(e.base)
        for idx in e.indices:
            if idx:
                self.visit(idx)
        for start, end in zip(e.starts, e.ends):
            self.visit(start)
            self.visit(end)

    def visit_TensorElement(self, e: TensorElement):
        self.visit(e.base)
        for idx in e.indices:
            self.visit(idx)

    def visit_Call(self, e: Call):
        self.visit(e.func_var)
        for arg in e.args:
            self.visit(arg)

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

    def visit_LessEqual(self, e: LessThan):
        return self.visit_Binary(e)

    def visit_Equal(self, e: Equal):
        return self.visit_Binary(e)

    def visit_And(self, e: And):
        return self.visit_Binary(e)

    def visit_Or(self, e: Or):
        return self.visit_Binary(e)

    def visit_Not(self, e: Not):
        a = self(e.a)
        if a is e.a:
            return e
        else:
            return Not(a)

    def visit_TensorSlice(self, e: TensorSlice):
        base = self(e.base)
        indices = [self(idx) if idx else None for idx in e.indices]
        starts = [self(v) for v in e.starts]
        ends = [self(v) for v in e.ends]
        if base is e.base and same_list(indices, e.indices) and same_list(starts, e.starts) and same_list(ends, e.ends):
            return e
        else:
            return TensorSlice(base, indices, starts, ends)

    def visit_TensorElement(self, e: TensorElement):
        base = self(e.base)
        indices = [self(idx) if idx else None for idx in e.indices]
        if base is e.base and same_list(indices, e.indices):
            return e
        else:
            return TensorElement(base, indices)

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
        axis = self(e.axis)
        shape = [self(v) for v in e.shape]
        if value is e.value and axis is e.axis and same_list(shape, e.shape):
            return e
        else:
            return ReduceCompute(value, shape, axis, e.reduce_type)

    def visit_AnyExpr(self, e: AnyExpr):
        return e

    def visit_ReduceComputePattern(self, e: ReduceComputePattern):
        return e

    def visit_TensorComputePattern(self, e: TensorComputePattern):
        return e

    def visit_ScalarExprPattern(self, e: ScalarExprPattern):
        return e


class StmtFunctor:
    def __init__(self):
        self.memo = {}

    def __call__(self, *args, **kwargs):
        return self.visit(*args, **kwargs)

    def visit(self, stmt: Stmt):
        if stmt in self.memo:
            return self.memo[stmt]
        if isinstance(stmt, EvaluateStmt):
            res = self.visit_EvaluateStmt(stmt)
        elif isinstance(stmt, BufferStoreStmt):
            res = self.visit_BufferStoreStmt(stmt)
        elif isinstance(stmt, AssignStmt):
            res = self.visit_AssignStmt(stmt)
        elif isinstance(stmt, LetStmt):
            res = self.visit_LetStmt(stmt)
        elif isinstance(stmt, ForStmt):
            res = self.visit_ForStmt(stmt)
        elif isinstance(stmt, IfStmt):
            res = self.visit_IfStmt(stmt)
        elif isinstance(stmt, AsmStmt):
            res = self.visit_AsmStmt(stmt)
        elif isinstance(stmt, AssertStmt):
            res = self.visit_AssertStmt(stmt)
        elif isinstance(stmt, BlackBoxStmt):
            res = self.visit_BlackBoxStmt(stmt)
        elif isinstance(stmt, SeqStmt):
            res = self.visit_SeqStmt(stmt)
        elif isinstance(stmt, Expr):
            res = self.visit_expr(stmt)
        else:
            raise ValueError()
        self.memo[stmt] = res
        return res

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

    def visit_AssertStmt(self, stmt: AssertStmt):
        raise NotImplementedError()

    def visit_AsmStmt(self, stmt: AsmStmt):
        raise NotImplementedError()

    def visit_BlackBoxStmt(self, stmt: BlackBoxStmt):
        raise NotImplementedError()

    def visit_SeqStmt(self, stmt: SeqStmt):
        raise NotImplementedError()


class StmtVisitor(StmtFunctor):
    def __init__(self):
        super().__init__()

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
        self.visit_expr(stmt.var)
        self.visit_expr(stmt.value)
        self.visit(stmt.body)

    def visit_ForStmt(self, stmt: ForStmt):
        self.visit_expr(stmt.loop_var)
        self.visit_expr(stmt.extent)
        self.visit(stmt.body)

    def visit_IfStmt(self, stmt: IfStmt):
        self.visit_expr(stmt.cond)
        self.visit(stmt.then_body)
        if stmt.else_body:
            self.visit(stmt.else_body)

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
    def __init__(self):
        super().__init__()

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
        var = self.visit_expr(stmt.var)
        value = self.visit_expr(stmt.value)
        body = self.visit(stmt.body)
        if var is stmt.var and value is stmt.value and body is stmt.body:
            return stmt
        else:
            return LetStmt(var, value, body)

    def visit_ForStmt(self, stmt: ForStmt):
        loop_var = self.visit_expr(stmt.loop_var)
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
    def __init__(self):
        ExprFunctor.__init__(self)
        StmtFunctor.__init__(self)

    def visit(self, obj):
        if isinstance(obj, Expr):
            return ExprFunctor.visit(self, obj)
        elif isinstance(obj, Stmt):
            return StmtFunctor.visit(self, obj)
        else:
            raise ValueError()

    def visit_expr(self, e: Expr):
        return self.visit(e)


class StmtExprVisitor(ExprVisitor, StmtVisitor):
    def __init__(self):
        ExprVisitor.__init__(self)
        StmtVisitor.__init__(self)

    def visit(self, obj):
        if isinstance(obj, Expr):
            return ExprVisitor.visit(self, obj)
        elif isinstance(obj, Stmt):
            return StmtVisitor.visit(self, obj)
        else:
            raise ValueError()

    def visit_expr(self, e: Expr):
        return self.visit(e)


class StmtExprRewriter(ExprRewriter, StmtRewriter):
    def __init__(self):
        ExprRewriter.__init__(self)
        StmtRewriter.__init__(self)

    def __call__(self, *args, **kwargs):
        return self.visit(*args, **kwargs)

    def visit(self, obj):
        if isinstance(obj, Expr):
            return ExprVisitor.visit(self, obj)
        elif isinstance(obj, Stmt):
            return StmtVisitor.visit(self, obj)
        else:
            raise ValueError()

    def visit_expr(self, e: Expr):
        return self.visit(e)


class TypeFunctor:
    def __init__(self):
        self.memo = {}

    def __call__(self, *args, **kwargs):
        return self.visit(*args, **kwargs)

    def visit(self, t: TypeNode):
        if t in self.memo:
            return self.memo[t]
        if isinstance(t, ScalarType):
            return self.visit_ScalarType(t)
        elif isinstance(t, TensorType):
            return self.visit_TensorType(t)
        elif isinstance(t, PointerType):
            return self.visit_PointerType(t)
        elif isinstance(t, ReferenceType):
            return self.visit_ReferenceType(t)
        elif isinstance(t, VoidType):
            return self.visit_VoidType(t)
        else:
            raise ValueError()

    def visit_ScalarType(self, t: ScalarType):
        raise NotImplementedError()

    def visit_TensorType(self, t: TensorType):
        raise NotImplementedError()

    def visit_PointerType(self, t: PointerType):
        raise NotImplementedError()

    def visit_ReferenceType(self, t: ReferenceType):
        raise NotImplementedError()

    def visit_VoidType(self, t: VoidType):
        raise NotImplementedError()


class WorkerFunctor:
    def __init__(self):
        self.memo = {}

    def __call__(self, *args, **kwargs):
        return self.visit(*args, **kwargs)

    def visit(self, worker: Worker):
        if worker in self.memo:
            return self.memo[worker]
        if isinstance(worker, Host):
            return self.visit_Host(worker)
        elif isinstance(worker, Grid):
            return self.visit_Grid(worker)
        elif isinstance(worker, ThreadBlock):
            return self.visit_ThreadBlock(worker)
        elif isinstance(worker, Warp):
            return self.visit_Warp(worker)
        elif isinstance(worker, Thread):
            return self.visit_Thread(worker)
        else:
            raise ValueError()

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


def same_list(lhs: List, rhs: List):
    if len(lhs) != len(rhs):
        return False
    return all(a is b for a, b in zip(lhs, rhs))
