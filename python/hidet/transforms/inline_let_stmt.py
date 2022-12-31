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
from collections import defaultdict

from hidet.ir.expr import Var, Expr, Constant, Add, Sub
from hidet.ir.functors import StmtExprRewriter, StmtExprVisitor, same_list
from hidet.ir.type import TensorType, TensorPointerType
from hidet.ir.stmt import Stmt, LetStmt
from hidet.transforms import Pass, FunctionBodyPass, RepeatFunctionPass


class LetVarRefAnalyzer(StmtExprVisitor):
    def __init__(self):
        super().__init__(use_memo=False)
        self.usage_count = None
        self.var2value = None

    def analyze(self, expr):
        self.usage_count = defaultdict(int)
        self.var2value = {}
        self.visit(expr)

    def visit(self, node):
        if isinstance(node, Var):
            self.usage_count[node] += 1
        return StmtExprVisitor.visit(self, node)

    def visit_LetStmt(self, stmt: LetStmt):
        for bind_var, bind_value in zip(stmt.bind_vars, stmt.bind_values):
            self.var2value[bind_var] = bind_value
            self.visit(bind_value)
        self.visit(stmt.body)


class NaiveLetStmtInlineRewriter(StmtExprRewriter):
    def __init__(self, inline_factor=1, inline_all=False):
        super().__init__()
        self.inline_factor = inline_factor
        self.inline_all = inline_all
        self.usage_count = None
        self.var2value = None

    def eliminate(self, stmt):
        self.memo.clear()
        # count the usage number and let var to its value
        analyzer = LetVarRefAnalyzer()
        analyzer.analyze(stmt)
        self.usage_count, self.var2value = analyzer.usage_count, analyzer.var2value
        # inline
        return self.visit(stmt)

    def should_inline(self, var: Var, expr: Expr) -> bool:
        if isinstance(var.type, (TensorPointerType, TensorType)):
            # do not inline tensor or tensor type
            return False
        if isinstance(expr, (Var, Constant)):
            # let v1 = v2
            # let v1 = constant
            return True
        elif self.usage_count[var] <= self.inline_factor or self.inline_all:
            # let v1 = expr and v1 is only used with in self.inline_factor times
            return True
        elif isinstance(expr, (Add, Sub)) and (isinstance(expr.a, Constant) or isinstance(expr.b, Constant)):
            # let v1 = expr + constant
            return True
        return False

    def visit_LetStmt(self, stmt: LetStmt):
        bind_vars = []
        bind_values = []
        for bind_var, bind_value in zip(stmt.bind_vars, stmt.bind_values):
            updated_value = self(bind_value)
            if self.should_inline(bind_var, updated_value):
                self.memo[bind_var] = updated_value
            else:
                bind_vars.append(bind_var)
                bind_values.append(updated_value)
        body = self(stmt.body)
        if same_list(bind_vars, stmt.bind_vars) and same_list(bind_values, stmt.bind_values) and body is stmt.body:
            return stmt
        else:
            if len(bind_vars) > 0:
                return LetStmt(bind_vars, bind_values, body)
            else:
                return body


class InlineNaiveLetStmtPass(FunctionBodyPass):
    def __init__(self, inline_factor=1, inline_all=False):
        super().__init__()
        self.inline_factor = inline_factor
        self.inline_all = inline_all

    def process_body(self, stmt: Stmt) -> Stmt:
        eliminator = NaiveLetStmtInlineRewriter(self.inline_factor, self.inline_all)
        return eliminator.eliminate(stmt)


def inline_let_stmt_pass(inline_factor=1, inline_all=False) -> Pass:
    if inline_all:
        return InlineNaiveLetStmtPass(inline_factor, inline_all)
    else:
        return RepeatFunctionPass(
            name='InlineLetStmtPass', passes=[InlineNaiveLetStmtPass(inline_factor, inline_all)], repeat_limit=10
        )
