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
from hidet.ir.expr import Expr, Call
from hidet.ir.type import FuncType
from hidet.ir.primitives.func import register_primitive_function, primitive_func_pool
from hidet.ir.primitives.math import MathFunctionSet, register_math_function_set


class CPUFloat32MathFunctionSet(MathFunctionSet):
    @staticmethod
    def register():
        unary_funcs = {
            'sin': 'sinf',
            'cos': 'cosf',
            'tan': 'tanf',
            'sinh': 'sinhf',
            'cosh': 'coshf',
            'tanh': 'tanhf',
            'asin': 'asinf',
            'acos': 'acosf',
            'atan': 'atanf',
            'asinh': 'asinhf',
            'acosh': 'acoshf',
            'atanh': 'atanhf',
            'exp': 'expf',
            'erf': 'erff',
            'sqrt': 'sqrtf',
            'rsqrt': 'rsqrtf',
            'log': 'logf',
            'round': 'roundf',
            'ceil': 'ceilf',
            'floor': 'floorf',
            'expm1': 'expm1f',
            'log2': 'log2f',
            'log10': 'log10f',
            'log1p': 'log1pf',
            'trunc': 'truncf',
            'isfinite': 'isfinite',
            'isinf': 'isinf',
            'isnan': 'isnan',
        }
        binary_funcs = {'min': 'fminf', 'max': 'fmaxf', 'pow': 'powf', 'mod': 'fmodf', 'atan2': 'atan2f'}
        ternary_funcs = {'fma': 'fmaf'}

        for name_map, num_args in zip([unary_funcs, binary_funcs, ternary_funcs], [1, 2, 3]):
            for name, codegen_name in name_map.items():
                register_primitive_function(
                    name='cpu_f32_{}'.format(name),
                    codegen_name=codegen_name,
                    func_or_type=FuncType(
                        param_types=['float32'] * num_args,
                        ret_type='float32' if name not in ['isfinite', 'isinf', 'isnan'] else 'bool',
                    ),
                )

    @staticmethod
    def call(name: str, *args) -> Expr:
        entry = primitive_func_pool.lookup_by_name('cpu_f32_{}'.format(name))
        return Call(entry.var, args)

    def sin(self, a: Expr) -> Expr:
        return self.call('sin', a)

    def cos(self, a: Expr) -> Expr:
        return self.call('cos', a)

    def tanh(self, a: Expr) -> Expr:
        return self.call('tanh', a)

    def exp(self, a: Expr) -> Expr:
        return self.call('exp', a)

    def erf(self, a: Expr) -> Expr:
        return self.call('erf', a)

    def sqrt(self, a: Expr) -> Expr:
        return self.call('sqrt', a)

    def rsqrt(self, a: Expr) -> Expr:
        return self.call('rsqrt', a)

    def log(self, a: Expr) -> Expr:
        return self.call('log', a)

    def round(self, a: Expr) -> Expr:
        return self.call('round', a)

    def ceil(self, a: Expr) -> Expr:
        return self.call('ceil', a)

    def floor(self, a: Expr) -> Expr:
        return self.call('floor', a)

    def tan(self, a: Expr) -> Expr:
        return self.call('tan', a)

    def sinh(self, a: Expr) -> Expr:
        return self.call('sinh', a)

    def cosh(self, a: Expr) -> Expr:
        return self.call('cosh', a)

    def asin(self, a: Expr) -> Expr:
        return self.call('asin', a)

    def acos(self, a: Expr) -> Expr:
        return self.call('acos', a)

    def atan(self, a: Expr) -> Expr:
        return self.call('atan', a)

    def asinh(self, a: Expr) -> Expr:
        return self.call('asinh', a)

    def acosh(self, a: Expr) -> Expr:
        return self.call('acosh', a)

    def atanh(self, a: Expr) -> Expr:
        return self.call('atanh', a)

    def expm1(self, a: Expr) -> Expr:
        return self.call('expm1', a)

    def log2(self, a: Expr) -> Expr:
        return self.call('log2', a)

    def log10(self, a: Expr) -> Expr:
        return self.call('log10', a)

    def log1p(self, a: Expr) -> Expr:
        return self.call('log1p', a)

    def trunc(self, a: Expr) -> Expr:
        return self.call('trunc', a)

    def isfinite(self, a: Expr) -> Expr:
        return self.call('isfinite', a)

    def isinf(self, a: Expr) -> Expr:
        return self.call('isinf', a)

    def isnan(self, a: Expr) -> Expr:
        return self.call('isnan', a)

    def min(self, a: Expr, b: Expr) -> Expr:
        return self.call('min', a, b)

    def max(self, a: Expr, b: Expr) -> Expr:
        return self.call('max', a, b)

    def pow(self, a: Expr, b: Expr) -> Expr:
        return self.call('pow', a, b)

    def mod(self, a: Expr, b: Expr) -> Expr:
        return self.call('mod', a, b)

    def atan2(self, a: Expr, b: Expr) -> Expr:
        return self.call('atan2', a, b)

    def fma(self, a: Expr, b: Expr, c: Expr) -> Expr:
        return self.call('fma', a, b, c)


cpu_f32_math_function_set = CPUFloat32MathFunctionSet()
cpu_f32_math_function_set.register()
register_math_function_set('cpu', 'float32', cpu_f32_math_function_set)
