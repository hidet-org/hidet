from __future__ import annotations
from typing import Any, Dict
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

    # constant creation
    def constant(self, value: Any) -> Constant:
        raise ValueError()

    def one(self) -> Constant:
        return self.constant(1)

    def zero(self) -> Constant:
        return self.constant(0)

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
        return NotImplemented

    def le(self, a: Expr, b: Expr) -> BoolExpr:
        return NotImplemented

    def gt(self, a: Expr, b: Expr) -> BoolExpr:
        return NotImplemented

    def ge(self, a: Expr, b: Expr) -> BoolExpr:
        return NotImplemented

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
        return self.one() / self.sqrt(a)

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


_registered_types: Dict[str, ScalarTypeAttr] = {}


def register_scalar_type(scalar_type: ScalarTypeAttr):
    name = scalar_type.name()
    if name in _registered_types:
        raise ValueError(f'Scalar type {name} has already been registered.')
    _registered_types[name] = scalar_type


def lookup_scalar_type(name: str) -> ScalarTypeAttr:
    return _registered_types[name]
