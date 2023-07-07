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
from typing import Union, List
from hidet.ir.expr import Expr
from hidet.ir.type import func_type
from hidet.ir.dtypes import float16, float16x2
from hidet.ir.primitives.func import register_primitive_function, primitive_func_pool
from hidet.ir.primitives.math import MathFunctionSet, register_math_function_set
import hidet.option


class CUDAFloat16x2MathFunctionSet(MathFunctionSet):
    # pylint: disable=abstract-method
    def register(self):
        entries = {
            'sin': ['h2sin', 1],
            'cos': ['h2cos', 1],
            'exp': ['h2exp', 1],
            'exp2': ['h2exp2', 1],
            'exp10': ['h2exp10', 1],
            'sqrt': ['h2sqrt', 1],
            'rsqrt': ['h2rsqrt', 1],
            'log': ['h2log', 1],
            'round': ['h2rint', 1],
            'ceil': ['h2ceil', 1],
            'floor': ['h2floor', 1],
            'min_sm80': ['__hmin2', 2],
            'max_sm80': ['__hmax2', 2],
            'fma': ['__hfma2', 3],
        }

        for name, (codegen_name, num_args) in entries.items():
            register_primitive_function(
                name='cuda_f16x2_{}'.format(name),
                codegen_name=codegen_name,
                func_or_type=func_type([float16] * num_args, float16),
            )
        register_primitive_function(
            name='cuda_f16x2_from_2xf16',
            func_or_type=func_type([float16, float16], float16x2),
            codegen_name='__halves2half2',
        )

    def call(self, name: str, *args) -> Expr:
        entry = primitive_func_pool.lookup_by_name(name)
        return entry.var(*args)

    def make_vector(self, items: Union[List[Expr], Expr]) -> Expr:
        if isinstance(items, Expr):
            items = [items]
        else:
            if not isinstance(items, (list, tuple)):
                raise ValueError('float16x2 requires a list of items')

        if len(items) == 1:
            return self.call('cuda_f16x2_from_f16', items[0])
        elif len(items) == 2:
            return self.call('cuda_f16x2_from_2xf16', items[0], items[1])
        else:
            raise ValueError('float16x2 requires 1 or 2 elements')

    def sin(self, a: Expr) -> Expr:
        return self.call('cuda_f16x2_sin', a)

    def cos(self, a: Expr) -> Expr:
        return self.call('cuda_f16x2_cos', a)

    def exp(self, a: Expr) -> Expr:
        return self.call('cuda_f16x2_exp', a)

    def sqrt(self, a: Expr) -> Expr:
        return self.call('cuda_f16x2_sqrt', a)

    def rsqrt(self, a: Expr) -> Expr:
        return self.call('cuda_f16x2_rsqrt', a)

    def log(self, a: Expr) -> Expr:
        return self.call('cuda_f16x2_log', a)

    def round(self, a: Expr) -> Expr:
        return self.call('cuda_f16x2_round', a)

    def ceil(self, a: Expr) -> Expr:
        return self.call('cuda_f16x2_ceil', a)

    def floor(self, a: Expr) -> Expr:
        return self.call('cuda_f16x2_floor', a)

    def min(self, a: Expr, b: Expr) -> Expr:
        if hidet.option.cuda.get_arch_pair() >= (8, 0):
            return self.call('cuda_f16x2_min_sm80', a, b)
        else:
            raise NotImplementedError('cuda_f16x2_min for < sm80 is not implemented')

    def max(self, a: Expr, b: Expr) -> Expr:
        if hidet.option.cuda.get_arch_pair() >= (8, 0):
            return self.call('cuda_f16x2_max_sm80', a, b)
        else:
            raise NotImplementedError('cuda_f16x2_max for < sm80 is not implemented')

    def fma(self, a: Expr, b: Expr, c: Expr) -> Expr:
        return self.call('cuda_f16x2_fma', a, b, c)


cuda_f16x2_math_function_set = CUDAFloat16x2MathFunctionSet()
cuda_f16x2_math_function_set.register()
register_math_function_set('cuda', 'float16x2', cuda_f16x2_math_function_set)
