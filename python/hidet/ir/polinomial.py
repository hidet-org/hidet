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
# pylint: disable=import-outside-toplevel, useless-parent-delegation, redefined-outer-name, redefined-builtin
# pylint: disable=useless-super-delegation, protected-access

from typing import Union, List, Dict
from hidet.ir.functors import ExprVisitor
from .node import Node
from .expr import Expr, Var, Constant, var

# Virtual `Var` used to represent bias of polinomial (`Poli`). Just small technical trick.
POLINOMIAL_BIAS_VAR = var('zzz_polinomial_bias')


# This Class represent a polynomial.
# It is used to simplify `Expr`.
class Poli(Node):
    def __init__(self, var: Var = POLINOMIAL_BIAS_VAR, coef: int = 0) -> None:
        super().__init__()
        if var is POLINOMIAL_BIAS_VAR:
            self.monos: dict[Var, int] = {var: coef}
        else:
            self.monos: dict[Var, int] = {var: coef, POLINOMIAL_BIAS_VAR: 0}

    def is_constant(self):
        self.remove_zeros()
        if len(self.monos) == 1 and POLINOMIAL_BIAS_VAR in self.monos:  # pylint: disable=simplifiable-if-statement
            return True
        else:
            return False

    def remove_zeros(self):
        copy = self.monos.copy()
        for key in copy:
            if self.monos[key] == 0 and key is not POLINOMIAL_BIAS_VAR:
                del self.monos[key]

    def get_bias(self):
        return self.monos.get(POLINOMIAL_BIAS_VAR, 0)

    # Convert to `Expr`.
    # terms_map is correspondence Var -> Expr
    # All vars from terms_map is changed to corresponding expresion. See also from_expr_to_poli
    def to_expr(self, terms_map: Dict[Var, Expr]) -> Expr:
        self.remove_zeros()
        expr = 0
        for var, coef in self.monos.items():
            if var is POLINOMIAL_BIAS_VAR:
                expr += coef
            elif var in terms_map:
                expr += terms_map[var] * coef
            else:
                expr += var * coef

        return expr

    @staticmethod
    def _binary_add(oper, a, b):
        if a is None or b is None:
            return None
        res_monos = {}
        all_vars = set(a.monos.keys()).union(set(b.monos.keys()))
        for key in all_vars:
            res_monos[key] = oper(a.monos.get(key, 0), b.monos.get(key, 0))
        res = Poli()
        res.monos = res_monos
        return res

    @staticmethod
    def _binary_mul(poli, const):
        if poli is None or const is None:
            return None
        assert isinstance(const, Poli)
        # Only linear polinomials with int coefs are supported now
        if not const.is_constant():
            return None
        const = const.get_bias()
        res_monos = {}
        for key in poli.monos.keys():
            res_monos[key] = poli.monos[key] * const
        res = Poli()
        res.monos = res_monos
        return res

    def __add__(self, other):
        return self._binary_add(lambda a, b: a + b, self, _convert(other))

    def __radd__(self, other):
        return self._binary_add(lambda a, b: a + b, _convert(other), self)

    def __sub__(self, other):
        return self._binary_add(lambda a, b: a - b, self, _convert(other))

    def __rsub__(self, other):
        return self._binary_add(lambda a, b: a - b, _convert(other), self)

    def __mul__(self, other):
        return self._binary_mul(self, other)

    def __rmul__(self, other):
        assert isinstance(other, int)
        return self._binary_mul(self, other)

    def __eq__(self, other):
        raise NotImplementedError("Equality comparison is not supported for this class.")

    def __repr__(self):
        res = ''
        sorted_monos = sorted(self.monos.items(), key=lambda x: x[0].hint)
        for mono in sorted_monos:
            res += f"{mono[1]}*{mono[0].hint}+"
        res = res.replace('*' + POLINOMIAL_BIAS_VAR.hint, '')
        res = res[:-1]
        return res

    def __str__(self):
        return self.__repr__()


def _convert(obj: Union[int, str, Var]) -> Poli:
    if isinstance(obj, Poli):
        return obj
    if isinstance(obj, int):
        return Poli(coef=obj)
    if isinstance(obj, str):
        return Poli(var=var(obj), coef=1)
    if isinstance(obj, Var):
        return Poli(var=obj, coef=1)
    # Can not convert. Return `None`. `None` is used as a indicator of impossibility of convertion
    return None


# Convert Expr to Poli.
# `attn_vars` is a list of Var in relation to which we constract polinomial.
# If we see sub expresion that is not polinomial and not contain attn_vars we changed it on
# virtual variable(term) and count it as a variable in polinomial.
# In terms_map we return correspondence between terms and expresions.
def from_expr_to_poli(expr: Expr, attn_vars: List[Var] = None) -> Poli:
    assert isinstance(expr, Expr)
    visitor = _Expr2PoliConverter(attn_vars)
    poli = visitor.visit(expr)
    terms_map = visitor.terms_map
    return poli, terms_map


class _Expr2PoliConverter(ExprVisitor):
    def __init__(self, attn_vars: List[Var]):
        self.terms_map: Dict[Var, Expr] = {}
        self.attn_vars = attn_vars
        super().__init__(use_memo=True)

    def visit_Constant(self, c: Constant):
        return Poli(coef=c.value)

    def visit_Var(self, v: Var):
        return Poli(var=v, coef=1)

    def _visit_binary(self, e: Expr, op):
        a = self.visit(e.a)
        b = self.visit(e.b)
        if a is None or b is None:
            return None
        return op(a, b)

    def visit_Add(self, e):
        return self._visit_binary(e, lambda x, y: x + y)

    def visit_Sub(self, e):
        return self._visit_binary(e, lambda x, y: x - y)

    def visit_Multiply(self, e):
        return self._visit_binary(e, lambda x, y: x * y)

    def visit_others(self, e):
        from hidet.ir.tools import collect_free_vars

        if self.attn_vars is None:
            return None

        vars = collect_free_vars(e)
        intersection = set(vars) & set(self.attn_vars)
        if len(intersection) == 0:
            term_var = var('term')
            self.terms_map[term_var] = e
            return Poli(var=term_var, coef=1)
        else:
            return None

    def visit_Mod(self, e):
        return self.visit_others(e)

    def visit_Div(self, e):
        return self.visit_others(e)
