from typing import Dict, Union, Mapping, Tuple, List
import sympy as S

from hidet.ir.type import ScalarType, TensorType
from hidet.ir.expr import Expr, Constant, Var, FloorDiv, Mod, Div, Multiply, Sub, Add, convert
from hidet.ir.dialects.compute import ScalarInput
from hidet.utils.namer import Namer

from .base import ExprFunctor


class HidetToSympyConverter(ExprFunctor):
    def __init__(self):
        super().__init__()
        self.namer = Namer()
        self.symbol_h2s: Dict[Expr, str] = {}

    def new_symbol(self, v: Union[Var, ScalarInput]):
        if v not in self.symbol_h2s:
            self.symbol_h2s[v] = self.namer.get_name(v)
        return S.Symbol(self.symbol_h2s[v])

    def visit_Add(self, e: Add):
        return self(e.a) + self(e.b)

    def visit_Sub(self, e: Sub):
        return self(e.a) - self(e.b)

    def visit_Multiply(self, e: Multiply):
        return self(e.a) * self(e.b)

    def visit_Div(self, e: Div):
        return self(e.a) / self(e.b)

    def visit_Mod(self, e: Mod):
        return self(e.a) % self(e.b)

    def visit_FloorDiv(self, e: FloorDiv):
        return self(e.a) / self(e.b)

    def visit_Var(self, e: Var):
        assert isinstance(e.type, ScalarType) and e.type.name == 'int32'
        return self.new_symbol(e)

    def visit_Constant(self, e: Constant):
        if isinstance(e.dtype, ScalarType):
            if e.dtype.name == 'int32':
                return S.Integer(e.value)
        # because we mainly use sympy to analyze the index expression, whose type is integer, we do not support other types
        raise NotImplementedError()

    def visit_ScalarInput(self, e: ScalarInput):
        return self.new_symbol(e)


class SympyToHidetConverter:
    def __init__(self, symbol_s2h: Mapping[S.Symbol, Expr]):
        self.memo = {}
        self.symbol_s2h = symbol_s2h

    def __call__(self, e: S.Expr):
        return self.visit(e)

    def visit(self, e: S.Expr):
        if e in self.memo:
            return self.memo[e]
        if isinstance(e, S.Symbol):
            ret = self.visit_Symbol(e)
        elif isinstance(e, S.Add):
            ret = self.visit_Add(e)
        elif isinstance(e, S.Mul):
            ret = self.visit_Mul(e)
        elif isinstance(e, S.Pow):
            ret = self.visit_Pow(e)
        elif isinstance(e, S.Integer):
            ret = self.visit_Integer(e)
        elif isinstance(e, S.Rational):
            ret = self.visit_Rational(e)
        elif isinstance(e, S.Mod):
            ret = self.visit_Mod(e)
        else:
            raise NotImplementedError()
        self.memo[e] = ret
        return ret

    def visit_Symbol(self, e: S.Symbol):
        if e.name not in self.symbol_s2h:
            raise ValueError(f'Symbol {e} not found in given mapping.')
        return self.symbol_s2h[e.name]

    def visit_Add(self, e: S.Add):
        args = [self(v) for v in e.args]
        assert len(args) > 0
        s = args[0]
        for v in args[1:]:
            if isinstance(v, Multiply) and isinstance(v.a, Constant) and v.a == -1:
                s = s - v.b
            else:
                s = s + v
        return s

    def visit_Mul(self, e: S.Mul):
        args = [self(v) for v in e.args]
        assert len(args) > 0
        s = args[0]
        for v in args[1:]:
            if isinstance(v, Multiply) and isinstance(v.a, Constant) and v.a == 1:
                s = s / v.b
            else:
                s = s * v
        return s

    def visit_Pow(self, e: S.Pow):
        args = [self(v) for v in e.args]
        assert len(args) == 2
        b, e = args
        assert isinstance(e, Constant) and e.dtype.name == 'int32'
        neg = False
        if e.value < 0:
            neg = True
            e.value = -e.value
        assert e.value >= 1
        s = b
        for i in range(e.value - 1):
            s = s * b
        if neg:
            s = convert(1) / b
        return s

    def visit_Integer(self, e: S.Integer):
        return convert(int(e.p))

    def visit_Rational(self, e: S.Rational):
        return convert(int(e.p)) / convert(int(e.q))

    def visit_Mod(self, e: S.Mod):
        args = [self(v) for v in e.args]
        assert len(args) == 2
        return args[0] % args[1]


def to_sympy(hidet_expr: Expr) -> Tuple[S.Expr, Mapping[str, Expr]]:
    converter = HidetToSympyConverter()
    sexpr = converter(hidet_expr)
    return sexpr, {a: b for b, a in converter.symbol_h2s.items()}


def from_sympy(sympy_expr: S.Expr, symbol_s2h: Mapping[str, Expr]):
    converter = SympyToHidetConverter(symbol_s2h)
    return converter(sympy_expr)


def equal(a: Expr, b: Expr):
    sexpr, symbol_map = to_sympy(a - b)
    return S.factor(sexpr) == 0


def coefficients(expr: Expr, bases: List[Union[Var, ScalarInput]]) -> Dict[Tuple, Expr]:
    """
    The given expr is a polynomial of given bases.
    This function returns the coefficients of each item in the polinomial.
    For example, let expr = k * i * j + 2 * i + 3 * j + 4 with bases = [i, j].
    We will return a dict:
    {(0, 0): 4, (0, 1): 3, (1, 0): 2, (1, 1): k}
    """
    sexpr, smap = to_sympy(expr)
    h2s_map = {b: a for a, b in smap.items()}
    for i, base in enumerate(bases):
        if base not in h2s_map:
            h2s_map[base] = f'_v{i}'
    spoly = S.Poly(sexpr, [S.Symbol(h2s_map[v]) for v in bases])
    s_dict = spoly.as_dict()
    h_dict = {}
    for item, s_expr in s_dict.items():
        h_dict[item] = from_sympy(s_expr, smap)
    return h_dict
