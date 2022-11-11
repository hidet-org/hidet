from hidet.ir.expr import Expr, Call
from hidet.ir.type import FuncType
from hidet.ir.func import Function
from hidet.ir.primitives.func import register_primitive_function, primitive_func_pool
from hidet.ir.primitives.math import MathFunctionSet, register_math_function_set


def cuda_f16_tanh_func() -> Function:
    from hidet.lang import script, f16, asm

    @script
    def cuda_f16_tanh(x: f16) -> f16:
        ret: f16 = f16(0.0)
        asm(template='tanh.approx.f16 %0, %1;', outputs=[ret], inputs=[x])
        return ret

    assert isinstance(cuda_f16_tanh, Function)
    return cuda_f16_tanh


class CUDAFloat16MathFunctionSet(MathFunctionSet):
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
                name='cuda_f16_{}'.format(name),
                codegen_name=codegen_name,
                func_or_type=FuncType(param_types=['float16'] * num_args, ret_type='float16'),
            )

        register_primitive_function(name='cuda_f16_tanh', func_or_type=cuda_f16_tanh_func())

    def call(self, name: str, *args) -> Expr:
        entry = primitive_func_pool.lookup_by_name(name)
        return Call(entry.var, args)

    def sin(self, a: Expr) -> Expr:
        return self.call('cuda_f16_sin', a)

    def cos(self, a: Expr) -> Expr:
        return self.call('cuda_f16_cos', a)

    def tanh(self, a: Expr) -> Expr:
        return self.call('cuda_f16_tanh', a)

    def exp(self, a: Expr) -> Expr:
        return self.call('cuda_f16_exp', a)

    def erf(self, a: Expr) -> Expr:
        raise ValueError('erf is not supported for float16 in cuda')

    def sqrt(self, a: Expr) -> Expr:
        return self.call('cuda_f16_sqrt', a)

    def rsqrt(self, a: Expr) -> Expr:
        return self.call('cuda_f16_rsqrt', a)

    def log(self, a: Expr) -> Expr:
        return self.call('cuda_f16_log', a)

    def round(self, a: Expr) -> Expr:
        return self.call('cuda_f16_round', a)

    def ceil(self, a: Expr) -> Expr:
        return self.call('cuda_f16_ceil', a)

    def floor(self, a: Expr) -> Expr:
        return self.call('cuda_f16_floor', a)

    def min(self, a: Expr, b: Expr) -> Expr:
        return self.call('cuda_f16_min', a, b)

    def max(self, a: Expr, b: Expr) -> Expr:
        return self.call('cuda_f16_max', a, b)

    def pow(self, a: Expr, b: Expr) -> Expr:
        raise ValueError('pow is not supported for float16 in cuda')

    def fma(self, a: Expr, b: Expr, c: Expr) -> Expr:
        return self.call('cuda_f16_fma', a, b, c)


cuda_f16_math_function_set = CUDAFloat16MathFunctionSet()
cuda_f16_math_function_set.register()
register_math_function_set('cuda', 'float16', cuda_f16_math_function_set)
