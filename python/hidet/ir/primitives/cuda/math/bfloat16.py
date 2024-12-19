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
from hidet.ir.dtypes import bfloat16, float32
from hidet.ir.primitives.func import register_primitive_function, primitive_func_pool
from hidet.ir.primitives.math import MathFunctionSet, register_math_function_set


class CUDABFloat16MathFunctionSet(MathFunctionSet):
    # pylint: disable=abstract-method
    def register(self):
        entries = {
            'sin': ['hsin', 1],
            'cos': ['hcos', 1],
            'exp': ['hexp', 1],
            'sqrt': ['hsqrt', 1],
            'rsqrt': ['hrsqrt', 1],
            'log': ['hlog', 1],
            'round': ['hrint', 1],
            'ceil': ['hceil', 1],
            'floor': ['hfloor', 1],
            'min': ['__hmin', 2],
            'max': ['__hmax', 2],
            'fma': ['__hfma', 3],
        }

        for name, (codegen_name, num_args) in entries.items():
            register_primitive_function(
                name='cuda_bf16_{}'.format(name),
                codegen_name=codegen_name,
                func_or_type=FuncType(param_types=['bfloat16'] * num_args, ret_type='bfloat16'),
            )

    def call(self, name: str, *args) -> Expr:
        entry = primitive_func_pool.lookup_by_name(name)
        return entry.var(*args)

    def sin(self, a: Expr) -> Expr:
        return self.call('cuda_bf16_sin', a)

    def cos(self, a: Expr) -> Expr:
        return self.call('cuda_bf16_cos', a)

    def tanh(self, a: Expr) -> Expr:
        from hidet.ir.expr import cast
        from hidet.ir.primitives.math import tanh

        return cast(tanh(cast(a, float32)), bfloat16)

    def exp(self, a: Expr) -> Expr:
        return self.call('cuda_bf16_exp', a)

    def erf(self, a: Expr) -> Expr:
        # use float32 erf to delegate the bfloat16 erf
        from hidet.ir.expr import cast
        from hidet.ir.primitives.math import erf

        return cast(erf(cast(a, float32)), bfloat16)

    def sqrt(self, a: Expr) -> Expr:
        return self.call('cuda_bf16_sqrt', a)

    def rsqrt(self, a: Expr) -> Expr:
        return self.call('cuda_bf16_rsqrt', a)

    def log(self, a: Expr) -> Expr:
        return self.call('cuda_bf16_log', a)

    def round(self, a: Expr) -> Expr:
        return self.call('cuda_bf16_round', a)

    def ceil(self, a: Expr) -> Expr:
        return self.call('cuda_bf16_ceil', a)

    def floor(self, a: Expr) -> Expr:
        return self.call('cuda_bf16_floor', a)

    def min(self, a: Expr, b: Expr) -> Expr:
        return self.call('cuda_bf16_min', a, b)

    def max(self, a: Expr, b: Expr) -> Expr:
        return self.call('cuda_bf16_max', a, b)

    def pow(self, a: Expr, b: Expr) -> Expr:
        # use float32 pow to delegate the bfloat16 pow
        from hidet.ir.expr import cast
        from hidet.ir.primitives.math import pow

        a = cast(a, float32)
        b = cast(b, float32)
        return cast(pow(a, b), bfloat16)

    def fma(self, a: Expr, b: Expr, c: Expr) -> Expr:
        return self.call('cuda_bf16_fma', a, b, c)


cuda_bf16_math_function_set = CUDABFloat16MathFunctionSet()
cuda_bf16_math_function_set.register()
register_math_function_set('cuda', 'bfloat16', cuda_bf16_math_function_set)
