from typing import List, Dict, Tuple, Optional
from hidet.ir.expr import Expr, Call
from hidet.ir.type import FuncType, DataType
from hidet.ir.utils.type_utils import numeric_promotion
from hidet.ir.primitives.func import register_primitive_function, lookup_primitive_function


class MathFunctionSet:
    # cast function
    def cast(self, a: Expr, cast_dtype: DataType) -> Optional[Expr]:
        """
        Cast expression a to cast_dtype.

        If the default implementation is used, return None. The default implementation is to use the cast expression
        in the underlying language (e.g. C cast):
          (cast_dtype)(a)

        Parameters
        ----------
        a: Expr
            The expression to be cast.

        cast_dtype: DataType
            The target data type.

        Returns
        -------
        ret:
            The cast expression. None if (cast_dtype)(a) is used to represent the cast.
        """
        return None

    # unary math functions
    def sin(self, a: Expr) -> Expr:
        raise NotImplementedError()

    def cos(self, a: Expr) -> Expr:
        raise NotImplementedError()

    def tanh(self, a: Expr) -> Expr:
        raise NotImplementedError()

    def exp(self, a: Expr) -> Expr:
        raise NotImplementedError()

    def erf(self, a: Expr) -> Expr:
        raise NotImplementedError()

    def sqrt(self, a: Expr) -> Expr:
        raise NotImplementedError()

    def rsqrt(self, a: Expr) -> Expr:
        raise NotImplementedError()

    def log(self, a: Expr) -> Expr:
        raise NotImplementedError()

    def round(self, a: Expr) -> Expr:
        raise NotImplementedError()

    def ceil(self, a: Expr) -> Expr:
        raise NotImplementedError()

    def floor(self, a: Expr) -> Expr:
        raise NotImplementedError()

    # binary math functions
    def min(self, a: Expr, b: Expr) -> Expr:
        raise NotImplementedError()

    def max(self, a: Expr, b: Expr) -> Expr:
        raise NotImplementedError()

    def pow(self, a: Expr, b: Expr) -> Expr:
        raise NotImplementedError()

    # ternary math functions
    def fma(self, a: Expr, b: Expr, c: Expr) -> Expr:
        raise NotImplementedError()


# (device, dtype) -> math function set
# such as ('cuda', 'float16') -> MathFunctionSet
registered_math_function_sets: Dict[Tuple[str, str], MathFunctionSet] = {}


def register_math_function_set(device: str, dtype: str, math_function_set: MathFunctionSet):
    if (device, dtype) in registered_math_function_sets:
        raise ValueError(f"Math function set for {device} and {dtype} already registered")
    registered_math_function_sets[(device, dtype)] = math_function_set


def type_infer_func(arg_types: List[DataType]) -> DataType:
    dtype = arg_types[0]
    for arg_type in arg_types[1:]:
        dtype = numeric_promotion(dtype, arg_type)
    return dtype


class MathFunctionSetGeneric(MathFunctionSet):
    def register(self):
        unary_names = ['sin', 'cos', 'tanh', 'exp', 'round', 'floor', 'ceil', 'rsqrt', 'sqrt', 'erf', 'log']
        binary_names = ['min', 'max', 'pow']
        ternary_names = ['fma']
        for name in unary_names + binary_names + ternary_names:
            register_primitive_function(
                name=f'generic_{name}',
                codegen_name=None,
                func_or_type=FuncType(type_infer_func=type_infer_func),
                generic=True,
            )

    def call(self, name, *args) -> Expr:
        entry = lookup_primitive_function(f'generic_{name}')
        return Call(entry.var, args)

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


generic_math_function_set = MathFunctionSetGeneric()
generic_math_function_set.register()


def sin(a: Expr) -> Expr:
    return generic_math_function_set.sin(a)


def cos(a: Expr) -> Expr:
    return generic_math_function_set.cos(a)


def tanh(a: Expr) -> Expr:
    return generic_math_function_set.tanh(a)


def exp(a: Expr) -> Expr:
    return generic_math_function_set.exp(a)


def erf(a: Expr) -> Expr:
    return generic_math_function_set.erf(a)


def sqrt(a: Expr) -> Expr:
    return generic_math_function_set.sqrt(a)


def rsqrt(a: Expr) -> Expr:
    return generic_math_function_set.rsqrt(a)


def log(a: Expr) -> Expr:
    return generic_math_function_set.log(a)


def round(a: Expr) -> Expr:
    return generic_math_function_set.round(a)


def ceil(a: Expr) -> Expr:
    return generic_math_function_set.ceil(a)


def floor(a: Expr) -> Expr:
    return generic_math_function_set.floor(a)


def min(a: Expr, b: Expr) -> Expr:
    return generic_math_function_set.min(a, b)


def max(a: Expr, b: Expr) -> Expr:
    return generic_math_function_set.max(a, b)


def pow(a: Expr, b: Expr) -> Expr:
    return generic_math_function_set.pow(a, b)


def fma(a: Expr, b: Expr, c: Expr) -> Expr:
    return generic_math_function_set.fma(a, b, c)
