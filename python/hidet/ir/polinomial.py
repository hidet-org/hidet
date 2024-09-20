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
from hidet.ir.functors import ExprVisitor
from .node import Node
from .expr import Expr, Var, Constant

POLINOMIAL_BIAS_NAME = 'zzz_polinomial_bias'

# Right now implemented Linear polinomial with integer coeficients only.
# TODO: expand functionality.
class Poli(Node):
    def __init__(self, var: str = POLINOMIAL_BIAS_NAME, coef: int = 0) -> None:
        super().__init__()
        # monos is a list of monomials of polinomial
        if var == POLINOMIAL_BIAS_NAME:
            self.monos: dict[str, int] = {var: coef}
        else:
            self.monos: dict[str, int] = {var: coef, POLINOMIAL_BIAS_NAME: 0}

    def is_constant(self):
        self.remove_zeros()
        if len(self.monos) == 1 and POLINOMIAL_BIAS_NAME in self.monos:  # pylint: disable=simplifiable-if-statement
            return True
        else:
            return False

    def remove_zeros(self):
        copy = self.monos.copy()
        for key in copy:
            if self.monos[key] == 0 and key != POLINOMIAL_BIAS_NAME:
                del self.monos[key]

    def get_bias(self):
        return self.monos.get(POLINOMIAL_BIAS_NAME, 0)

    @staticmethod
    def _binary_add(oper, a, b):
        if a is None or b is None:
            return None
        res_monos = {}
        all_keys = set(a.monos.keys()).union(set(b.monos.keys()))
        for key in all_keys:
            res_monos[key] = oper(a.monos.get(key, 0), b.monos.get(key, 0))
        res = Poli()
        res.monos = res_monos
        return res

    @staticmethod
    def _binary_mul(a, b: int):
        if a is None or b is None:
            return None
        # Only linear polinomials with int coefs are supported now
        if not isinstance(b, int):
            return None
        res_monos = {}
        for key in a.monos.keys():
            res_monos[key] = a.monos[key] * b
        res = Poli()
        res.monos = res_monos
        return res

    @staticmethod
    def _convert(obj):
        if isinstance(obj, Poli):
            return obj
        if isinstance(obj, int):
            return Poli(coef=obj)
        if isinstance(obj, str):
            return Poli(var=obj, coef=1)
        if isinstance(obj, Var):
            return Poli(var=obj.name, coef=1)
        # Cannot convert. Return None
        return None

    def __add__(self, other):
        return self._binary_add(lambda a, b: a + b, self, self._convert(other))

    def __radd__(self, other):
        return self._binary_add(lambda a, b: a + b, self._convert(other), self)

    def __sub__(self, other):
        return self._binary_add(lambda a, b: a - b, self, self._convert(other))

    def __rsub__(self, other):
        return self._binary_add(lambda a, b: a - b, self._convert(other), self)

    def __mul__(self, other):
        return self._binary_mul(self, other)

    def __rmul__(self, other):
        assert isinstance(other, int)
        return self._binary_mul(self, other)

    def __eq__(self, other):
        assert False

    def __str__(self):
        res = ''
        sorted_keys = sorted(self.monos.keys())
        for key in sorted_keys:
            res += f"{self.monos[key]}*{key}+"
        res = res.replace('*' + POLINOMIAL_BIAS_NAME, '')
        res = res[:-1]
        return res


# Convert expression `Expr` to polinomial `Poli`.
#
# Return:
#   None if conversion isn't successful
def from_expr_to_poli(expr: Expr) -> Poli:
    assert isinstance(expr, Expr)
    visitor = Expr2PoliConverter()
    poli = visitor.visit(expr)
    return poli


class Expr2PoliConverter(ExprVisitor):
    def __init__(self):
        super().__init__(use_memo=True)
        self.poli: Poli = Poli()

    def visit_Constant(self, c: Constant):
        return c.value

    def visit_Var(self, v: Var):
        return Poli(v.hint, coef=1)

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
