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
from typing import Callable
from hidet.ir.expr import Expr, ExprInt64, ExprFloat16
from hidet.ir.type import FuncType, DataType
from hidet.ir.func import Function
from hidet.ir.dtypes import float16, float32, int64
from hidet.ir.primitives.func import register_primitive_function, primitive_func_pool, call_primitive_func
from hidet.ir.primitives.math import MathFunctionSet, register_math_function_set
from hidet.utils import initialize


@initialize()
def register_float16_primitives():
    register_primitive_function('hip_i64_to_f16', FuncType([int64], float16), codegen_name='__ll2half_rn')


def hip_i64_to_f16(a: ExprInt64) -> ExprFloat16:
    return call_primitive_func('hip_i64_to_f16', [a])


class HIPFloat16MathFunctionSet(MathFunctionSet):
    # pylint: disable=abstract-method
    def register(self):
        entries = {
            # 'abs': ['__habs', 1],
            'sin': ['hsin', 1],
            'cos': ['hcos', 1],
            'exp': ['hexp', 1],
            'sqrt': ['hsqrt', 1],
            # 'rsqrt': ['hrsqrt', 1],
            'log': ['hlog', 1],
            'round': ['hrint', 1],
            'ceil': ['hceil', 1],
            'floor': ['hfloor', 1],
            # 'min_sm80': ['__hmin', 2],
            # 'max_sm80': ['__hmax', 2],
            'fma': ['__hfma', 3],
        }

        for name, (codegen_name, num_args) in entries.items():
            register_primitive_function(
                name='hip_f16_{}'.format(name),
                codegen_name=codegen_name,
                func_or_type=FuncType(param_types=['float16'] * num_args, ret_type='float16'),
            )

        from hidet.ir.primitives.math import tanh, abs

        self.register_via_delegate('min', float16, float32, min, 2)
        self.register_via_delegate('max', float16, float32, max, 2)
        self.register_via_delegate('abs', float16, float32, abs, 1)
        self.register_via_delegate('rsqrt', float16, float32, lambda x: 1.0 / self.sqrt(x), 1)
        self.register_via_delegate('tanh', float16, float32, tanh, 1)

    def register_via_delegate(
        self, name: str, target_type: DataType, delegate_type: DataType, delegate: Callable, num_args: int
    ):
        from hidet.lang import script, cast, attrs

        if num_args == 1:

            @script
            def delegated_primitive(v: target_type) -> target_type:
                attrs.func_name = 'hip_f16_{}'.format(name)
                attrs.func_kind = 'hip_internal'
                return cast(delegate(cast(v, delegate_type)), target_type)

        elif num_args == 2:

            @script
            def delegated_primitive(a: target_type, b: target_type) -> target_type:
                attrs.func_name = 'hip_f16_{}'.format(name)
                attrs.func_kind = 'hip_internal'
                return cast(delegate(cast(a, delegate_type), cast(b, delegate_type)), target_type)

        elif num_args == 3:

            @script
            def delegated_primitive(a: target_type, b: target_type, c: target_type) -> target_type:
                attrs.func_name = 'hip_f16_{}'.format(name)
                attrs.func_kind = 'hip_internal'
                return cast(
                    delegate(cast(a, delegate_type), cast(b, delegate_type), cast(c, delegate_type)), target_type
                )

        else:
            raise ValueError('Unsupported num_args: {}'.format(num_args))

        assert isinstance(delegated_primitive, Function)
        register_primitive_function(name='hip_f16_{}'.format(name), func_or_type=delegated_primitive)

    def call(self, name: str, *args) -> Expr:
        entry = primitive_func_pool.lookup_by_name(name)
        return entry.var(*args)

    def abs(self, a: Expr) -> Expr:
        return self.call('hip_f16_abs', a)

    def sin(self, a: Expr) -> Expr:
        return self.call('hip_f16_sin', a)

    def cos(self, a: Expr) -> Expr:
        return self.call('hip_f16_cos', a)

    def tanh(self, a: Expr) -> Expr:
        return self.call('hip_f16_tanh', a)

    def exp(self, a: Expr) -> Expr:
        return self.call('hip_f16_exp', a)

    def erf(self, a: Expr) -> Expr:
        # use float32 erf to delegate the float16 erf
        from hidet.ir.expr import cast
        from hidet.ir.primitives.math import erf

        return cast(erf(cast(a, float32)), float16)

    def sqrt(self, a: Expr) -> Expr:
        return self.call('hip_f16_sqrt', a)

    def rsqrt(self, a: Expr) -> Expr:
        return self.call('hip_f16_rsqrt', a)

    def log(self, a: Expr) -> Expr:
        return self.call('hip_f16_log', a)

    def round(self, a: Expr) -> Expr:
        return self.call('hip_f16_round', a)

    def ceil(self, a: Expr) -> Expr:
        return self.call('hip_f16_ceil', a)

    def floor(self, a: Expr) -> Expr:
        return self.call('hip_f16_floor', a)

    def min(self, a: Expr, b: Expr) -> Expr:
        return self.call('hip_f16_min', a, b)

    def max(self, a: Expr, b: Expr) -> Expr:
        return self.call('hip_f16_max', a, b)

    def pow(self, a: Expr, b: Expr) -> Expr:
        # use float32 pow to delegate the float16 pow
        from hidet.ir.expr import cast
        from hidet.ir.primitives.math import pow

        a = cast(a, float32)
        b = cast(b, float32)
        return cast(pow(a, b), float16)

    def fma(self, a: Expr, b: Expr, c: Expr) -> Expr:
        return self.call('hip_f16_fma', a, b, c)


hip_f16_math_function_set = HIPFloat16MathFunctionSet()
hip_f16_math_function_set.register()
register_math_function_set('hip', 'float16', hip_f16_math_function_set)
