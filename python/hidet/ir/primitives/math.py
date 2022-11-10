from hidet.ir.expr import Expr


class MathFunctionSet:
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
        raise NotImplementedError()

    def log(self, a: Expr) -> Expr:
        raise NotImplementedError()

    def ceil(self, a: Expr) -> Expr:
        raise NotImplementedError()

    def floor(self, a: Expr) -> Expr:
        raise NotImplementedError()

    # binary math functions
    def min(self, a: Expr, b: Expr) -> Expr:
        raise NotImplementedError()

    def max(self, a: Expr, b: Expr) -> Expr:
        raise NotImplementedError()

    def pow(self, a: Expr, b: Expr) -> Expr:
        raise NotImplementedError()

    # ternary math functions
    def fma(self, a: Expr, b: Expr, c: Expr) -> Expr:
        return a * b + c
