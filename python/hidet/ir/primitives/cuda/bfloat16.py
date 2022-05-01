from hidet.utils import initialize
from ..func import FuncType, register_primitive_function, primitive_func_pool
from .funcs import register_unary_dialect_primitive_function, register_binary_dialect_primitive_function
from hidet.ir.primitives.base.funcs import erf, tanh, pow


@initialize()
def register_primitive_functions_bfloat16():
    unary_names = [
        '__hneg', 'hsin', 'hcos', 'hexp', 'hrint', 'hfloor', 'hceil', 'hrsqrt', 'hsqrt',
    ]
    binary_names = [
        '__hmin', '__hmax',
    ]
    ternary_names = [
        '__hfma',
    ]
    base2bfloat16 = {
        'neg': '__hneg',
        'sin': 'hsin',
        'cos': 'hcos',
        'exp': 'hexp',
        'round': 'hrint',
        'floor': 'hceil',
        'ceil': 'hceil',
        'rsqrt': 'hrsqrt',
        'sqrt': 'hsqrt',
        'min': '__hmin',
        'max': '__hmax',

        # cuda c does not provide the following functions, we use f16 -> f32 -> f -> f16 path
        'tanh': 'htanh',
        'erf': 'herf',
        'pow': 'hpow'
    }
    for unary in unary_names:
        register_primitive_function('bfloat16', unary, FuncType(param_types=['bfloat16'], ret_type='bfloat16'))
    for binary in binary_names:
        register_primitive_function('bfloat16', binary, FuncType(param_types=['bfloat16', 'bfloat16'], ret_type='bfloat16'))
    for ternary in ternary_names:
        register_primitive_function('bfloat16', ternary, FuncType(param_types=['bfloat16', 'bfloat16', 'bfloat16'], ret_type='bfloat16'))

    register_unary_dialect_primitive_function(space='bfloat16', func_name='htanh', generic_func=tanh, target_dtype='bfloat16', dialect_dtype='float32')
    register_unary_dialect_primitive_function(space='bfloat16', func_name='herf', generic_func=erf, target_dtype='bfloat16', dialect_dtype='float32')
    register_binary_dialect_primitive_function(space='bfloat16', func_name='hpow', generic_func=pow, target_dtype='bfloat16', dialect_dtype='float32')
    for base_name, bf16_name in base2bfloat16.items():
        primitive_func_pool.lookup_by_name('base', base_name).dispatch_dtype(dtype='bfloat16', space='bfloat16', func_name=bf16_name)
