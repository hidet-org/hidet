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
from typing import List, Dict, Optional, ContextManager

from hidet.ir.type import FuncType, data_type
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

    def __init__(self, stack, scope_stmt):
        self.stack: 'ScopeStack' = stack
        self.scope_stmt = scope_stmt
        self.level = None
        self.parent: Optional['Scope'] = None
        self.declare_vars: List[Var] = []
        self.defined_vars: List[Var] = []
        self.var2value: Dict[Var, Optional[Expr]] = {}
        self.defined_predicates: List[List[Expr]] = []
        self.predicate_vars: List[Var] = []

    def __enter__(self):
        scopes = self.stack.scopes
        self.parent = scopes[0] if len(scopes) > 0 else None
        self.level = len(scopes)
        scopes.append(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        scope = self.stack.scopes.pop()
        assert scope is self

    def declare(self, var: Var):
        # declare a variable at current scope
        self.declare_vars.append(var)
        self.var2value[var] = None
        assert var not in self.stack.var2scope
        self.stack.var2scope[var] = self

    def define(self, var: Var, value: Expr):
        self.defined_vars.append(var)
        self.var2value[var] = value
        assert var not in self.stack.var2scope
        self.stack.var2scope[var] = self

    def define_predicate(self, predicate: Expr) -> Expr:
        if len(self.defined_predicates) == 0 or len(self.defined_predicates[-1]) == 32:
            var = Var('p', type=data_type('uint32'))
            self.defined_predicates.append([])
            self.predicate_vars.append(var)
            self.stack.var2scope[var] = self
        self.defined_predicates[-1].append(predicate)
        mask = 1 << (len(self.defined_predicates[-1]) - 1)
        return BitwiseAnd(self.predicate_vars[-1], mask)

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

    def find_scope_for_expr(self, expr) -> 'Scope':
        used_vars = collect(expr, Var)
        levels = [self.var2scope[used_var].level for used_var in used_vars if not isinstance(used_var.type, FuncType)]
        max_level = max(levels)
        return self.scopes[max_level]

    def new_scope(self, scope_stmt=None):
        return Scope(self, scope_stmt)

    def current(self) -> Scope:
        assert len(self.scopes) > 0
        return self.scopes[-1]


class FuncStmtExprRewriterWithScope(FuncStmtExprRewriter):
    def __init__(self, use_memo=False):
        super().__init__(use_memo=use_memo)
        self.scope_stack = ScopeStack()

    def new_scope(self, stmt=None) -> ContextManager[Scope]:
        return self.scope_stack.new_scope(stmt)

    def scope_to_define(self, expr: Expr) -> Scope:
        return self.scope_stack.find_scope_for_expr(expr)

    def visit_Function(self, func: Function):
        with self.new_scope(None) as scope:
            for extern_var in func.extern_vars:
                scope.declare(extern_var)
            for param in func.params:
                scope.declare(param)
            body = scope.wrap(self.visit(func.body))
            return Function(
                func.name,
                func.params,
                body,
                func.ret_type,
                kind=func.kind,
                extern_vars=func.extern_vars,
                attrs=func.attrs,
            )

    def visit_ForStmt(self, stmt: ForStmt):
        with self.new_scope(stmt) as scope:
            self.visit(stmt.extent)
            scope.declare(stmt.loop_var)
            body = scope.wrap(self.visit(stmt.body))
            return ForStmt(stmt.loop_var, stmt.extent, stmt.unroll, body)

    def visit_LetStmt(self, stmt: LetStmt):
        with self.new_scope(stmt) as scope:
            for var, value in zip(stmt.bind_vars, stmt.bind_values):
                scope.define(var, self.visit(value))
            return scope.wrap(self.visit(stmt.body))
