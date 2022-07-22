from hidet.utils import initialize
from ..func import FuncType, register_primitive_function
from ..func import primitive_func_pool as pool


@initialize()
def register_primitive_functions_int32():
    binary_names = [
        'max', 'min'
    ]
    base2int32 = {
        'max': 'max',
        'min': 'min'
    }
    for name in binary_names:
        register_primitive_function(
            name='{}_{}'.format('int32', name),
            codegen_name=name,
            func_or_type=FuncType(param_types=['int32', 'int32'], ret_type='int32'),
            generic=False
        )
    for a, b in base2int32.items():
        base_name = '{}_{}'.format('base', a)
        int32_name = '{}_{}'.format('int32', b)
        pool.lookup_by_name(base_name).dispatch_dtype(dtype='int32', dispatched_func_name=int32_name)
