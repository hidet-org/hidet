from typing import Mapping
from collections import defaultdict

from hidet.ir.expr import Var, Expr, Constant, Mod
from hidet.ir.functors import StmtExprRewriter, StmtExprVisitor, rewrite
from hidet.ir.stmt import Stmt, LetStmt
from hidet.transforms import Pass, FunctionBodyPass, RepeatFunctionPass


class LetVarRefAnalyzer(StmtExprVisitor):
    def __init__(self):
        super().__init__()
        self.usage_count = None
        self.var2value = None

    def analyze(self, expr):
        self.usage_count = defaultdict(int)
        self.var2value = {}
        self.visit(expr)

    def visit(self, obj):
        if isinstance(obj, Var):
            self.usage_count[obj] += 1
        return StmtExprVisitor.visit(self, obj)

    def visit_LetStmt(self, stmt: LetStmt):
        self.var2value[stmt.var] = stmt.value
        # do not visit stmt.var because we are counting usage
        self.visit(stmt.value)
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

    def visit_LetStmt(self, stmt: LetStmt):
        # Inline in the following cases
        #    case 1 usage count <= inline factor: replace reference var to its value expression
        #    case 2 value is a Constant or Var
        if self.usage_count[stmt.var] <= self.inline_factor or self.inline_all:  # case 1
            self.memo[stmt.var] = self(stmt.value)
            return self(stmt.body)
        elif isinstance(stmt.value, (Var, Constant)):  # case 2
            self.memo[stmt.var] = self(stmt.value)
            return self(stmt.body)
        else:
            value = self.visit(stmt.value)
            body = self.visit(stmt.body)
            if body is stmt.body and value is stmt.value:
                return stmt
            else:
                return LetStmt(stmt.var, value, body)


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
            name='InlineLetStmtPass',
            passes=[
                InlineNaiveLetStmtPass(inline_factor, inline_all)
            ],
            repeat_limit=10)
