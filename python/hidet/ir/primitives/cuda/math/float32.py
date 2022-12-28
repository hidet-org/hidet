from hidet.ir.expr import Expr, Call
from hidet.ir.type import FuncType
from hidet.ir.primitives.func import register_primitive_function, primitive_func_pool
from hidet.ir.primitives.math import MathFunctionSet, register_math_function_set


class CUDAFloat32MathFunctionSet(MathFunctionSet):
    # pylint: disable=abstract-method
    def register(self):
        entries = {
            'sin': ['sinf', 1],
            'cos': ['cosf', 1],
            'tanh': ['tanhf', 1],
            'exp': ['expf', 1],
            'erf': ['erff', 1],
            'sqrt': ['sqrtf', 1],
            'rsqrt': ['rsqrtf', 1],
            'log': ['logf', 1],
            'round': ['roundf', 1],
            'ceil': ['ceilf', 1],
            'floor': ['floorf', 1],
            'min': ['fminf', 2],
            'max': ['fmaxf', 2],
            'pow': ['powf', 2],
            'fma': ['fmaf', 3],
        }

        for name, (codegen_name, num_args) in entries.items():
            register_primitive_function(
                name='cuda_f32_{}'.format(name),
                codegen_name=codegen_name,
                func_or_type=FuncType(param_types=['float32'] * num_args, ret_type='float32'),
            )

    def call(self, name: str, *args) -> Expr:
        entry = primitive_func_pool.lookup_by_name(name)
        return Call(entry.var, args)

    def sin(self, a: Expr) -> Expr:
        return self.call('cuda_f32_sin', a)

    def cos(self, a: Expr) -> Expr:
        return self.call('cuda_f32_cos', a)

    def tanh(self, a: Expr) -> Expr:
        return self.call('cuda_f32_tanh', a)

    def exp(self, a: Expr) -> Expr:
        return self.call('cuda_f32_exp', a)

    def erf(self, a: Expr) -> Expr:
        return self.call('cuda_f32_erf', a)

    def sqrt(self, a: Expr) -> Expr:
        return self.call('cuda_f32_sqrt', a)

    def rsqrt(self, a: Expr) -> Expr:
        return self.call('cuda_f32_rsqrt', a)

    def log(self, a: Expr) -> Expr:
        return self.call('cuda_f32_log', a)

    def round(self, a: Expr) -> Expr:
        return self.call('cuda_f32_round', a)

    def ceil(self, a: Expr) -> Expr:
        return self.call('cuda_f32_ceil', a)

    def floor(self, a: Expr) -> Expr:
        return self.call('cuda_f32_floor', a)

    def min(self, a: Expr, b: Expr) -> Expr:
        return self.call('cuda_f32_min', a, b)

    def max(self, a: Expr, b: Expr) -> Expr:
        return self.call('cuda_f32_max', a, b)

    def pow(self, a: Expr, b: Expr) -> Expr:
        return self.call('cuda_f32_pow', a, b)

    def fma(self, a: Expr, b: Expr, c: Expr) -> Expr:
        return self.call('cuda_f32_fma', a, b, c)


cuda_f32_math_function_set = CUDAFloat32MathFunctionSet()
cuda_f32_math_function_set.register()
register_math_function_set('cuda', 'float32', cuda_f32_math_function_set)
