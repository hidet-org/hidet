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

from hidet.ir.primitives import blockIdx, threadIdx
from hidet.ir.type import TensorPointerType, TensorType, ArrayType, FuncType
from hidet.ir.expr import Var, Expr, Constant, Add, Sub, Call
from hidet.ir.functors import IRRewriter, IRVisitor
from hidet.ir.stmt import LetStmt
from hidet.ir.func import Function
from hidet.ir.tools import collect
from hidet.transforms import Pass, FunctionPass, RepeatFunctionPass
from hidet.utils import same_list


class LetVarRefAnalyzer(IRVisitor):
    def __init__(self):
        super().__init__(use_memo=False)
        self.usage_count = None
        self.var2value = None

    def analyze(self, stmt):
        self.usage_count = defaultdict(int)
        self.var2value = {}
        self.visit(stmt)

    def visit(self, node):
        if isinstance(node, Var):
            self.usage_count[node] += 1
        return IRVisitor.visit(self, node)

    def visit_LetStmt(self, stmt: LetStmt):
        for bind_var, bind_value in zip(stmt.bind_vars, stmt.bind_values):
            self.var2value[bind_var] = bind_value
            self.visit(bind_value)
        self.visit(stmt.body)


class NaiveLetStmtInlineRewriter(IRRewriter):
    thread_constant_vars = [
        blockIdx.x,
        blockIdx.y,
        blockIdx.z,  # block index
        threadIdx.x,
        threadIdx.y,
        threadIdx.z,  # thread index
    ]

    def __init__(self, inline_factor=1, inline_all=False):
        super().__init__()
        self.inline_factor = inline_factor
        self.inline_all = inline_all
        self.usage_count = None
        self.var2value = None

    def eliminate(self, node):
        self.memo.clear()
        # count the usage number and let var to its value
        analyzer = LetVarRefAnalyzer()
        analyzer.analyze(node)
        self.usage_count, self.var2value = analyzer.usage_count, analyzer.var2value
        # inline
        return self.visit(node)

    def has_side_effect(self, expr):

        return len(collect(expr, Call)) > 0

    def should_inline(self, var: Var, expr: Expr) -> bool:
        from hidet.ir.tools import ExprHash

        if isinstance(var.type, (TensorPointerType, TensorType)):
            tt_type = var.type.tensor_type if isinstance(var.type, TensorPointerType) else var.type
            assert isinstance(tt_type, TensorType)
            if isinstance(expr, Var):
                if isinstance(expr.type, (TensorPointerType, TensorType)):
                    tt_type2 = expr.type.tensor_type if isinstance(expr.type, TensorPointerType) else expr.type
                    assert isinstance(tt_type2, TensorType)
                    return ExprHash().hash(tt_type) == ExprHash().hash(tt_type2)
                return True
            elif isinstance(expr, Constant):
                return not isinstance(expr.type, (TensorType, ArrayType))
            else:
                return False
        elif isinstance(expr, Var) and expr in self.var2value:
            # let v1 = v2 # and v2 is also bound by a let statement
            return True
        elif isinstance(expr, Constant):
            # let v1 = constant
            return not isinstance(expr.type, (TensorType, ArrayType))
        elif isinstance(expr, (Add, Sub)) and (
            (isinstance(expr.a, Constant) and expr.b in self.var2value)
            or (isinstance(expr.b, Constant) and expr.a in self.var2value)
        ):
            # let v1 = v2 + constant  # and v2 is also bound by a let statement
            return True
        elif self.usage_count[var] <= self.inline_factor or self.inline_all:
            # let v1 = expr
            # 1. v1 is only used with in self.inline_factor times
            # 2. all variables in expr are bound by let statements or are constants regards the thread
            # 3. expr has no side effects
            # 4. var is not a function variable
            if isinstance(var.type, FuncType) or self.has_side_effect(expr):
                return False
            else:
                used_vars = collect(expr, Var)
                for used_var in used_vars:
                    if used_var not in self.thread_constant_vars and used_var not in self.var2value:
                        return False
                return True
        else:
            return False

    def visit_LetStmt(self, stmt: LetStmt):
        bind_vars = []
        bind_values = []
        for bind_var, bind_value in zip(stmt.bind_vars, stmt.bind_values):
            updated_var = self(bind_var)
            updated_value = self(bind_value)
            if self.should_inline(updated_var, updated_value):
                self.memo[bind_var] = updated_value
            else:
                bind_vars.append(updated_var)
                bind_values.append(updated_value)
        body = self(stmt.body)
        if same_list(bind_vars, stmt.bind_vars) and same_list(bind_values, stmt.bind_values) and body is stmt.body:
            return stmt
        else:
            if len(bind_vars) > 0:
                return LetStmt(bind_vars, bind_values, body)
            else:
                return body


class InlineNaiveLetStmtPass(FunctionPass):
    def __init__(self, inline_factor=1, inline_all=False):
        super().__init__()
        self.inline_factor = inline_factor
        self.inline_all = inline_all

    def process_func(self, func: Function) -> Function:
        eliminator = NaiveLetStmtInlineRewriter(self.inline_factor, self.inline_all)
        return eliminator.eliminate(func)


def inline_let_stmt_pass(inline_factor=1, inline_all=False) -> Pass:
    if inline_all:
        return InlineNaiveLetStmtPass(inline_factor, inline_all)
    else:
        return RepeatFunctionPass(
            name='InlineLetStmtPass', passes=[InlineNaiveLetStmtPass(inline_factor, inline_all)], repeat_limit=10
        )
