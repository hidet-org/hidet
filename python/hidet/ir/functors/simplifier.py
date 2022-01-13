from typing import Union
import operator
from hidet.ir.expr import Expr, BinaryOp, Add, Sub, Multiply, Div, Mod, FloorDiv, LessThan, LessEqual, Equal, Constant, And, Or, Not
from hidet.ir.expr import is_one, is_zero, is_true, is_false, convert
from hidet.ir.stmt import Stmt, IfStmt, SeqStmt
from hidet.ir.functors import StmtExprRewriter, same_list


class Simplifier(StmtExprRewriter):
    def visit_Binary(self, e: BinaryOp):
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
        elif isinstance(e, And):
            if is_false(a) or is_false(b):
                return convert(False)
            if is_true(a):
                return b
            if is_true(b):
                return a
        elif isinstance(e, Or):
            if is_true(a) or is_true(b):
                return convert(True)
            if is_false(a):
                return b
            if is_false(b):
                return a
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
                Equal: operator.eq
            }
            if e.__class__ in op_dict:
                if a.dtype.name == 'int32' and b.dtype.name == 'int32' and isinstance(e, Div):
                    # the Div for int32 will use floordiv. Override the native behavior of python
                    return a.value // b.value
                else:
                    return convert(op_dict[e.__class__](a.value, b.value))
            elif isinstance(e, And):
                return convert(a.value and b.value)
            elif isinstance(e, Or):
                return convert(a.value or b.value)
            else:
                raise ValueError()
        if a is e.a and b is e.b:
            return e
        return e.__class__(a, b)

    def visit_Not(self, e: Not):
        a = self(e.a)
        if isinstance(a, Constant):
            return convert(not a.value)
        if a is e.a:
            return e
        else:
            return Not(a)


def simplify(node: Union[Stmt, Expr]):
    simplifier = Simplifier()
    return simplifier(node)


def equal(a: Expr, b: Expr):
    pass

