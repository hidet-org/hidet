from typing import ContextManager, List, Dict, Optional
from hidet.ir.expr import Let, Expr, Var
from hidet.ir.stmt import Stmt, SeqStmt, LetStmt, ForStmt
from hidet.ir.func import Function
from hidet.ir.functors import FuncStmtExprRewriter, collect
from hidet.ir.task import Grid
from hidet.ir.primitives import thread_idx, block_idx
from .base import FunctionPass


class Scope:
    """
    Every variable (i.e., parameter variable, local variable, loop variable, let variable) much be declared or defined
    in a scope. Parameter, local and loop variable should be declared, because we should not move it place. Every
    let variable should be defined (with their value).
    """

    def __init__(self, stack, level, parent):
        self.stack = stack
        self.level = level
        self.parent: 'Scope' = parent
        self.declare_vars: List[Var] = []
        self.defined_vars: List[Var] = []
        self.var2value: Dict[Var, Optional[Expr]] = {}

    def declare(self, var: Var):
        self.declare_vars.append(var)
        self.var2value[var] = None
        assert var not in self.stack.var2scope
        self.stack.var2scope[var] = self

    def define(self, var: Var, value: Expr):
        # find the inner-most (with largest level) scope that contains the used var to define
        used_vars = collect(value, Var)
        levels = [self.stack.var2scope[used_var].level for used_var in used_vars]
        max_level = max(levels)
        scope = self
        while max_level < scope.level:
            scope = scope.parent
        scope.defined_vars.append(var)
        scope.var2value[var] = value
        assert var not in self.stack.var2scope
        self.stack.var2scope[var] = scope

    def wrap(self, body):
        if len(self.defined_vars) > 0:
            bind_vars = self.defined_vars
            bind_values = [self.var2value[var] for var in bind_vars]
            ret = LetStmt(bind_vars, bind_values, body)
        else:
            ret = body
        for var in self.defined_vars + self.declare_vars:
            del self.stack.var2scope[var]
        return ret


class ScopeStack:
    def __init__(self):
        self.scopes = []
        self.var2scope: Dict[Var, Scope] = {}

    def __enter__(self) -> Scope:
        parent = self.scopes[-1] if len(self.scopes) > 0 else None
        level = len(self.scopes)
        scope = Scope(self, level, parent)
        self.scopes.append(scope)
        return scope

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.scopes.pop()


class UpliftLetStmtRewriter(FuncStmtExprRewriter):
    def __init__(self):
        super().__init__(use_memo=False)
        self.stack = ScopeStack()

    def new_scope(self) -> ContextManager[Scope]:
        return self.stack

    def visit_Function(self, func: Function):
        with self.new_scope() as scope:
            for extern_var in func.extern_vars:
                scope.declare(extern_var)
            for param in func.params:
                scope.declare(param)
            for local_var in func.local_vars:
                scope.declare(local_var)
            body = scope.wrap(self.visit(func.body))
            return Function(func.name, func.params, body, func.ret_type, func.local_vars, func.extern_vars, func.attrs)

    def visit_ForStmt(self, stmt: ForStmt):
        with self.new_scope() as scope:
            self.visit(stmt.extent)
            scope.declare(stmt.loop_var)
            body = scope.wrap(self.visit(stmt.body))
            return ForStmt(stmt.loop_var, stmt.extent, stmt.unroll, body)

    def visit_LetStmt(self, stmt: LetStmt):
        with self.new_scope() as scope:
            for var, value in zip(stmt.bind_vars, stmt.bind_values):
                scope.define(var, self.visit(value))
            return scope.wrap(self.visit(stmt.body))

    def visit_Let(self, e: Let):
        raise ValueError('Please run expand_let_expr pass first')


class UpliftLetStmtPass(FunctionPass):
    def process_func(self, func: Function) -> Function:
        rewriter = UpliftLetStmtRewriter()
        return rewriter.visit(func)


def uplift_let_stmt_pass():
    return UpliftLetStmtPass()
