from hidet.utils import initialize
from hidet.ir.primitives.base.generic import erf, tanh, pow
from ..func import FuncType, register_primitive_function, primitive_func_pool
from .funcs import register_unary_dialect_primitive_function, register_binary_dialect_primitive_function


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
    for codegen_names, num_args in zip([unary_names, binary_names, ternary_names], [1, 2, 3]):
        func_type = FuncType(param_types=['bfloat16'] * num_args, ret_type='bfloat16')
        for codegen_name in codegen_names:
            name = '{}_{}'.format('bfloat16', codegen_name)
            register_primitive_function(name=name, func_or_type=func_type, codegen_name=codegen_name)
    register_unary_dialect_primitive_function(func_name='bfloat16_htanh', generic_func=tanh, target_dtype='bfloat16',
                                              dialect_dtype='float32')
    register_unary_dialect_primitive_function(func_name='bfloat16_herf', generic_func=erf, target_dtype='bfloat16',
                                              dialect_dtype='float32')
    register_binary_dialect_primitive_function(func_name='bfloat16_hpow', generic_func=pow, target_dtype='bfloat16',
                                               dialect_dtype='float32')

    for a, b in base2bfloat16.items():
        base_name = '{}_{}'.format('base', a)
        bf16_name = '{}_{}'.format('bfloat16', b)
        primitive_func_pool.lookup_by_name(base_name).dispatch_dtype(dtype='bfloat16', dispatched_func_name=bf16_name)
