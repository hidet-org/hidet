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


class CUDAComplex128MathFunctionSet(MathFunctionSet):
    # pylint: disable=abstract-method
    @staticmethod
    def register():
        unary_funcs = {'sin': 'sin', 'cos': 'cos', 'abs': 'abs', 'exp': 'exp'}

        for name_map, num_args in zip([unary_funcs], [1]):
            for name, codegen_name in name_map.items():
                register_primitive_function(
                    name='cuda_c128_{}'.format(name),
                    codegen_name=codegen_name,
                    func_or_type=FuncType(
                        param_types=['complex128'] * num_args,
                        ret_type='complex128' if name not in ['abs'] else 'float64',
                    ),
                )

    @staticmethod
    def call(name: str, *args) -> Expr:
        entry = primitive_func_pool.lookup_by_name('cuda_c128_{}'.format(name))
        return entry.var(*args)

    def sin(self, a: Expr) -> Expr:
        return self.call('sin', a)

    def cos(self, a: Expr) -> Expr:
        return self.call('cos', a)

    def exp(self, a: Expr) -> Expr:
        return self.call('exp', a)


cuda_c128_math_function_set = CUDAComplex128MathFunctionSet()
cuda_c128_math_function_set.register()
register_math_function_set('cuda', 'complex128', cuda_c128_math_function_set)
