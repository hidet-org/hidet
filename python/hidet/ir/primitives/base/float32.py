from hidet.utils import initialize
from ..func import FuncType, register_primitive_function
from ..func import primitive_func_pool as pool


@initialize()
def register_primitive_functions_float32():
    unary_names = [
        'sinf', 'cosf', 'tanhf', 'expf', 'roundf', 'floorf', 'ceilf', 'rsqrtf', 'sqrtf', 'erff', 'logf'
    ]
    binary_names = [
        'fminf', 'fmaxf', 'powf'
    ]
    ternary_names = [
        'fmaf'
    ]
    base2float32 = {
        'sin': 'sinf',
        'cos': 'cosf',
        'tanh': 'tanhf',
        'exp': 'expf',
        'round': 'roundf',
        'floor': 'floorf',
        'ceil': 'ceilf',
        'rsqrt': 'rsqrtf',
        'erf': 'erff',
        'log': 'logf',
        'sqrt': 'sqrtf',
        'min': 'fminf',
        'max': 'fmaxf',
        'pow': 'powf',
        'fma': 'fmaf'
    }
    for names, param_types in zip([unary_names, binary_names, ternary_names], [['float32'], ['float32'] * 2, ['float32'] * 3]):
        for name in names:
            register_primitive_function(
                name='{}_{}'.format('cuda_fp32', name),
                codegen_name=name,
                func_or_type=FuncType(param_types=param_types, ret_type='float32'),
                generic=False
            )
    for a, b in base2float32.items():
        base_name = '{}_{}'.format('base', a)
        fp32_name = '{}_{}'.format('cuda_fp32', b)
        pool.lookup_by_name(base_name).dispatch_dtype(dtype='float32', dispatched_func_name=fp32_name)
