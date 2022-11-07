from hidet.ir.expr import Expr, Constant

# just a hint
BoolExpr = Expr


class ScalarTypeAttr:
    def name(self) -> str:
        raise NotImplementedError()

    def short_name(self) -> str:
        raise NotImplementedError()

    # properties
    def nbytes(self) -> int:
        raise NotImplementedError()

    def is_float(self) -> bool:
        raise NotImplementedError()

    def is_integer(self) -> bool:
        raise NotImplementedError()

    def is_vector(self) -> bool:
        raise NotImplementedError()

    # basic arithmetic operations
    def add(self, a: Expr, b: Expr) -> Expr:
        return NotImplemented

    def sub(self, a: Expr, b: Expr) -> Expr:
        return NotImplemented

    def mul(self, a: Expr, b: Expr) -> Expr:
        return NotImplemented

    def div(self, a: Expr, b: Expr) -> Expr:
        return NotImplemented

    def mod(self, a: Expr, b: Expr) -> Expr:
        return NotImplemented

    # comparison operations
    def eq(self, a: Expr, b: Expr) -> BoolExpr:
        return NotImplemented

    def ne(self, a: Expr, b: Expr) -> BoolExpr:
        return NotImplemented

    def lt(self, a: Expr, b: Expr) -> BoolExpr:
        raise NotImplementedError()

    def le(self, a: Expr, b: Expr) -> BoolExpr:
        raise NotImplementedError()

    def gt(self, a: Expr, b: Expr) -> BoolExpr:
        raise NotImplementedError()

    def ge(self, a: Expr, b: Expr) -> BoolExpr:
        raise NotImplementedError()

    # math functions

    # unary math functions
    def sin(self, a: Expr) -> Expr:
        raise NotImplementedError()

    def cos(self, a: Expr) -> Expr:
        raise NotImplementedError()

    def tanh(self, a: Expr) -> Expr:
        raise NotImplementedError()

    def exp(self, a: Expr) -> Expr:
        raise NotImplementedError()

    def erf(self, a: Expr) -> Expr:
        raise NotImplementedError()

    def sqrt(self, a: Expr) -> Expr:
        raise NotImplementedError()

    def rsqrt(self, a: Expr) -> Expr:
        return NotImplemented

    def min(self, a: Expr, b: Expr) -> Expr:
        raise NotImplementedError()

    def max(self, a: Expr, b: Expr) -> Expr:
        raise NotImplementedError()

    def fma(self, a: Expr, b: Expr, c: Expr) -> Expr:
        return NotImplemented
