from typing import List, Dict, Optional, ContextManager

from hidet.ir.type import ScalarType
from hidet.ir.expr import Expr, Var, BitwiseAnd, LeftShift, BitwiseOr
from hidet.ir.functors import collect
from hidet.ir.stmt import LetStmt, ForStmt
from hidet.ir.func import Function
from hidet.ir.functors import FuncStmtExprRewriter


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
        self.defined_predicates: List[List[Expr]] = []
        self.predicate_vars: List[Var] = []

    def find_scope_for_expr(self, expr) -> 'Scope':
        used_vars = collect(expr, Var)
        levels = [self.stack.var2scope[used_var].level for used_var in used_vars]
        max_level = max(levels)
        scope = self
        while max_level < scope.level:
            scope = scope.parent
        return scope

    def declare(self, var: Var):
        # declare a variable at current scope
        self.declare_vars.append(var)
        self.var2value[var] = None
        assert var not in self.stack.var2scope
        self.stack.var2scope[var] = self

    def define(self, var: Var, value: Expr, at_current_scope=False):
        # define a variable at the outer-most scope with given value
        # find the outer-most (with minimal level) scope that contains the used var to define
        if at_current_scope:
            scope = self
        else:
            scope = self.find_scope_for_expr(value)
        scope.defined_vars.append(var)
        scope.var2value[var] = value
        assert var not in self.stack.var2scope
        self.stack.var2scope[var] = scope

    def define_predicate(self, predicate: Expr, at_current_scope=False) -> Expr:
        if at_current_scope:
            scope = self
        else:
            scope = self.find_scope_for_expr(predicate)
        if len(scope.defined_predicates) == 0 or len(scope.defined_predicates[-1]) == 32:
            var = Var('p', type=ScalarType('uint32'))
            scope.defined_predicates.append([])
            scope.predicate_vars.append(var)
            self.stack.var2scope[var] = scope
        scope.defined_predicates[-1].append(predicate)
        # mask = LeftShift(1, len(scope.defined_predicates[-1]) - 1)
        mask = 1 << (len(scope.defined_predicates[-1]) - 1)
        return BitwiseAnd(scope.predicate_vars[-1], mask)

    def wrap(self, body):
        # wrap the body with defined variables at current scope
        bind_vars = self.defined_vars
        bind_values = [self.var2value[var] for var in bind_vars]
        for p_var, p_exprs in zip(self.predicate_vars, self.defined_predicates):
            bind_vars.append(p_var)
            bind_values.append(BitwiseOr.join_list([LeftShift(p, idx) for idx, p in enumerate(p_exprs)]))
        if len(bind_vars) > 0:
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

    def current(self) -> Scope:
        assert len(self.scopes) > 0
        return self.scopes[-1]


class FuncStmtExprRewriterWithScope(FuncStmtExprRewriter):
    def __init__(self, use_memo=False):
        super().__init__(use_memo=use_memo)
        self.scope_stack = ScopeStack()

    def new_scope(self) -> ContextManager[Scope]:
        return self.scope_stack

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
                scope.define(var, self.visit(value), at_current_scope=True)
            return scope.wrap(self.visit(stmt.body))
