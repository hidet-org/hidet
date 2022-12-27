from hidet.ir.expr import Expr, Call, ExprInt64, ExprFloat16, ExprInt16
from hidet.ir.type import FuncType
from hidet.ir.func import Function
from hidet.ir.dtypes import int16
from hidet.ir.primitives.func import register_primitive_function, primitive_func_pool, call_primitive_func
from hidet.ir.primitives.math import MathFunctionSet, register_math_function_set
from hidet.utils import initialize


@initialize()
def register_float16_primitives():
    from hidet.ir.dtypes import int64, float16

    register_primitive_function('cuda_i64_to_f16', FuncType([int64], float16), codegen_name='__ll2half_rn')


def cuda_i64_to_f16(a: ExprInt64) -> ExprFloat16:
    return call_primitive_func('cuda_i64_to_f16', [a])


def reinterpret_f16_as_u16(a: ExprFloat16) -> ExprInt16:
    """
    Reinterpret a float16 as an int16.

    Equivalent c: (*(short*)(&a)) where a is a float16.

    Why we need this transformation?

    In CUDA, cuda_fp16.h provides a class __half to represent float16. Hidet directly uses it as the type of float16.

    However, inside cuda_fp16.h, this class is defined as a struct with a short member, which prevents us from using
    it directly in an embedded ptx assemble. Thus, we need to reinterpret the memory of __half as an int16 before
    using it in an embedded ptx assemble.

    Parameters
    ----------
    a: ExprFloat16
        The float16 to reinterpret.

    Returns
    -------
    ret: ExprInt16
        The int16 expression that reinterpret the memory of float16 input data.
    """
    from hidet.ir.expr import address, cast, deref

    return deref(cast(address(a), ~int16))


class CUDAFloat16MathFunctionSet(MathFunctionSet):
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
                name='cuda_f16_{}'.format(name),
                codegen_name=codegen_name,
                func_or_type=FuncType(param_types=['float16'] * num_args, ret_type='float16'),
            )

        self._register_tanh()

    def _register_tanh(self):
        from hidet.lang import script, f16, asm

        @script
        def cuda_f16_tanh(x: f16) -> f16:
            ret: f16 = f16(0.0)
            asm(
                template='tanh.approx.f16 %0, %1;',
                outputs=[reinterpret_f16_as_u16(ret)],
                inputs=[reinterpret_f16_as_u16(x)],
            )
            return ret

        assert isinstance(cuda_f16_tanh, Function)

        register_primitive_function(name='cuda_f16_tanh', func_or_type=cuda_f16_tanh)

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
        # use float32 erf to delegate the float16 erf
        from hidet.ir.expr import cast
        from hidet.ir.dtypes import float32, float16
        from hidet.ir.primitives.math import erf

        return cast(erf(cast(a, float32)), float16)

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
        # use float32 pow to delegate the float16 pow
        from hidet.ir.expr import cast
        from hidet.ir.dtypes import float32, float16
        from hidet.ir.primitives.math import pow

        a = cast(a, float32)
        b = cast(b, float32)
        return cast(pow(a, b), float16)

    def fma(self, a: Expr, b: Expr, c: Expr) -> Expr:
        return self.call('cuda_f16_fma', a, b, c)


cuda_f16_math_function_set = CUDAFloat16MathFunctionSet()
cuda_f16_math_function_set.register()
register_math_function_set('cuda', 'float16', cuda_f16_math_function_set)
