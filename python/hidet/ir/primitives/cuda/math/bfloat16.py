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
from hidet.ir.func import Function
from hidet.ir.primitives.func import register_primitive_function, primitive_func_pool
from hidet.ir.primitives.math import MathFunctionSet, register_math_function_set


def cuda_bf16_tanh_func() -> Function:
    from hidet.lang import script, bf16, asm

    @script
    def cuda_bf16_tanh(x: bf16) -> bf16:
        ret: bf16 = bf16(0.0)
        asm(template='tanh.approx.bf16 %0, %1;', outputs=[ret], inputs=[x])
        return ret

    assert isinstance(cuda_bf16_tanh, Function)
    return cuda_bf16_tanh


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

        register_primitive_function(name='cuda_bf16_tanh', func_or_type=cuda_bf16_tanh_func())

    def call(self, name: str, *args) -> Expr:
        entry = primitive_func_pool.lookup_by_name(name)
        return Call(entry.var, args)

    def sin(self, a: Expr) -> Expr:
        return self.call('cuda_bf16_sin', a)

    def cos(self, a: Expr) -> Expr:
        return self.call('cuda_bf16_cos', a)

    def tanh(self, a: Expr) -> Expr:
        return self.call('cuda_bf16_tanh', a)

    def exp(self, a: Expr) -> Expr:
        return self.call('cuda_bf16_exp', a)

    def erf(self, a: Expr) -> Expr:
        raise ValueError('erf is not supported for bfloat16 in cuda')

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
        raise ValueError('pow is not supported for bfloat16 in cuda')

    def fma(self, a: Expr, b: Expr, c: Expr) -> Expr:
        return self.call('cuda_bf16_fma', a, b, c)


cuda_bf16_math_function_set = CUDABFloat16MathFunctionSet()
cuda_bf16_math_function_set.register()
register_math_function_set('cuda', 'bfloat16', cuda_bf16_math_function_set)
