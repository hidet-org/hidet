from hidet.ir.expr import Expr, Call
from hidet.ir.type import FuncType
from hidet.ir.primitives.func import register_primitive_function, primitive_func_pool
from hidet.ir.primitives.math import MathFunctionSet, register_math_function_set


class CUDAInt64MathFunctionSet(MathFunctionSet):
    def register(self):
        entries = {'min': ['min', 2], 'max': ['max', 2]}

        for name, (codegen_name, num_args) in entries.items():
            register_primitive_function(
                name='cuda_i64_{}'.format(name),
                codegen_name=codegen_name,
                func_or_type=FuncType(param_types=['int64'] * num_args, ret_type='int64'),
            )

    def call(self, name: str, *args) -> Expr:
        entry = primitive_func_pool.lookup_by_name(name)
        return Call(entry.var, args)

    def min(self, a: Expr, b: Expr) -> Expr:
        return self.call('cuda_i64_min', a, b)

    def max(self, a: Expr, b: Expr) -> Expr:
        return self.call('cuda_i64_max', a, b)

    def sin(self, a: Expr) -> Expr:
        raise ValueError('sin is not supported for int64')

    def cos(self, a: Expr) -> Expr:
        raise ValueError('cos is not supported for int64')

    def tanh(self, a: Expr) -> Expr:
        raise ValueError('tanh is not supported for int64')

    def exp(self, a: Expr) -> Expr:
        raise ValueError('exp is not supported for int64')

    def erf(self, a: Expr) -> Expr:
        raise ValueError('erf is not supported for int64')

    def sqrt(self, a: Expr) -> Expr:
        raise ValueError('sqrt is not supported for int64')

    def rsqrt(self, a: Expr) -> Expr:
        raise ValueError('rsqrt is not supported for int64')

    def log(self, a: Expr) -> Expr:
        raise ValueError('log is not supported for int64')

    def round(self, a: Expr) -> Expr:
        raise ValueError('round is not supported for int64')

    def ceil(self, a: Expr) -> Expr:
        raise ValueError('ceil is not supported for int64')

    def floor(self, a: Expr) -> Expr:
        raise ValueError('floor is not supported for int64')

    def pow(self, a: Expr, b: Expr) -> Expr:
        raise ValueError('pow is not supported for int64')

    def fma(self, a: Expr, b: Expr, c: Expr) -> Expr:
        raise ValueError('fma is not supported for int64')


cuda_i64_math_function_set = CUDAInt64MathFunctionSet()
cuda_i64_math_function_set.register()
register_math_function_set('cuda', 'int64', cuda_i64_math_function_set)
