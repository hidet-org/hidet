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
from typing import Union
import operator
from hidet.ir.expr import (
    Expr,
    BinaryOp,
    Add,
    Sub,
    Multiply,
    Div,
    Mod,
    FloorDiv,
    LessThan,
    LessEqual,
    Equal,
    Constant,
    BitwiseAnd,
    BitwiseOr,
    BitwiseXor,
)
from hidet.ir.expr import LogicalAnd, LogicalOr, LogicalNot, is_one, is_zero, is_true, is_false, convert
from hidet.ir.stmt import Stmt, IfStmt, SeqStmt, ForStmt
from hidet.ir.functors import StmtExprRewriter, rewrite


class Simplifier(StmtExprRewriter):
    def visit_Binary(self, e: BinaryOp):  # pylint: disable=too-many-branches
        a = self(e.a)
        b = self(e.b)
        if isinstance(e, Add):
            if is_zero(a):
                return b
            if is_zero(b):
                return a
        elif isinstance(e, Sub):
            if is_zero(b):
                return a
        elif isinstance(e, Multiply):
            if is_one(a):
                return b
            if is_one(b):
                return a
            if is_zero(a) or is_zero(b):
                return convert(0)
        elif isinstance(e, Div):
            if is_one(b):
                return a
        elif isinstance(e, Mod):
            if is_one(e.b):
                return convert(0)
        elif isinstance(e, FloorDiv):
            if is_one(b):
                return a
        elif isinstance(e, LessThan):
            pass
        elif isinstance(e, LessEqual):
            pass
        elif isinstance(e, Equal):
            pass
        elif isinstance(e, LogicalAnd):
            if is_false(a) or is_false(b):
                return convert(False)
            if is_true(a):
                return b
            if is_true(b):
                return a
        elif isinstance(e, LogicalOr):
            if is_true(a) or is_true(b):
                return convert(True)
            if is_false(a):
                return b
            if is_false(b):
                return a
        elif isinstance(e, BitwiseAnd):
            pass
        elif isinstance(e, BitwiseOr):
            pass
        elif isinstance(e, BitwiseXor):
            pass
        else:
            raise ValueError()

        if isinstance(a, Constant) and isinstance(b, Constant):
            op_dict = {
                Add: operator.add,
                Sub: operator.sub,
                Multiply: operator.mul,
                Div: operator.truediv,
                Mod: operator.mod,
                FloorDiv: operator.floordiv,
                LessThan: operator.lt,
                Equal: operator.eq,
            }
            if e.__class__ in op_dict:
                if a.type.name == 'int32' and b.type.name == 'int32' and isinstance(e, Div):
                    # the Div for int32 will use floordiv. Override the native behavior of python
                    return convert(a.value // b.value, 'int32')
                else:
                    return convert(op_dict[e.__class__](a.value, b.value))
            elif isinstance(e, LogicalAnd):
                return convert(a.value and b.value)
            elif isinstance(e, LogicalOr):
                return convert(a.value or b.value)
            else:
                raise ValueError()
        if a is e.a and b is e.b:
            return e
        return e.__class__(a, b)

    def visit_Not(self, e: LogicalNot):
        a = self(e.a)
        if isinstance(a, Constant):
            return convert(not a.value)
        if a is e.a:
            return e
        else:
            return LogicalNot(a)

    def visit_IfStmt(self, stmt: IfStmt):
        cond = self.visit_expr(stmt.cond)
        then_body = self.visit(stmt.then_body)
        else_body = self.visit(stmt.else_body) if stmt.else_body else None
        if is_true(cond):
            return then_body
        elif is_false(cond):
            if else_body:
                return else_body
            else:
                return SeqStmt([])
        else:
            if cond is stmt.cond and then_body is stmt.then_body and else_body is stmt.else_body:
                return stmt
            else:
                return IfStmt(cond, then_body, else_body)

    def visit_ForStmt(self, stmt: ForStmt):
        loop_var = self(stmt.loop_var)
        extent = self(stmt.extent)
        body = self(stmt.body)
        if is_one(extent):
            return rewrite(stmt.body, {loop_var: convert(0)})
        else:
            if loop_var is stmt.loop_var and body is stmt.body:
                return stmt
            else:
                return ForStmt(loop_var, extent, stmt.unroll, body)


def simplify(node: Union[Stmt, Expr], repeat_limit=10):
    if isinstance(node, (int, float)):
        return node
    simplifier = Simplifier()
    for _ in range(repeat_limit):
        old_node = node
        node = simplifier(node)
        if old_node is node:
            break
    return node


def simplify_to_int(node: Union[Expr, int], repeat_limit=10) -> int:
    if isinstance(node, int):
        return node
    node = simplify(node, repeat_limit)
    assert isinstance(node, Constant) and node.type.name in [
        'int32',
        'uint8',
    ], 'Invalid input. Input type must be one of [int32, uint8].'
    return node.value
