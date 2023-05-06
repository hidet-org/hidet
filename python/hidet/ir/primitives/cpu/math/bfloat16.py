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
from hidet.ir.expr import Expr
from hidet.ir.type import FuncType
from hidet.ir.primitives.func import register_primitive_function, primitive_func_pool
from hidet.ir.primitives.math import MathFunctionSet, register_math_function_set


class CPUBFloat16MathFunctionSet(MathFunctionSet):
    # pylint: disable=abstract-method
    def register(self):
        entries = {
            'sin': ['sin', 1],
            'cos': ['cos', 1],
            'exp': ['exp', 1],
            'sqrt': ['sqrt', 1],
            'rsqrt': ['rsqrt', 1],
            'log': ['log', 1],
            'round': ['round', 1],
            'ceil': ['ceil', 1],
            'floor': ['floor', 1],
            'tanh': ['tanh', 1],
            'erf': ['erf', 1],
            'min': ['min', 2],
            'max': ['max', 2],
            'pow': ['pow', 2],
            'fma': ['fma', 3],
        }

        for name, (codegen_name, num_args) in entries.items():
            register_primitive_function(
                name='cpu_bf16_{}'.format(name),
                codegen_name=codegen_name,
                func_or_type=FuncType(param_types=['bfloat16'] * num_args, ret_type='bfloat16'),
            )

    def call(self, name: str, *args) -> Expr:
        entry = primitive_func_pool.lookup_by_name('cpu_bf16_{}'.format(name))
        return entry.var(*args)

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

    def min(self, a: Expr, b: Expr) -> Expr:
        return self.call('min', a, b)

    def max(self, a: Expr, b: Expr) -> Expr:
        return self.call('max', a, b)

    def pow(self, a: Expr, b: Expr) -> Expr:
        return self.call('pow', a, b)

    def fma(self, a: Expr, b: Expr, c: Expr) -> Expr:
        return self.call('fma', a, b, c)


cpu_bf16_math_function_set = CPUBFloat16MathFunctionSet()
cpu_bf16_math_function_set.register()
register_math_function_set('cpu', 'bfloat16', cpu_bf16_math_function_set)
