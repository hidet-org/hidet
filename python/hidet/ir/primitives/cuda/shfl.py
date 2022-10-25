from hidet.ir.primitives.func import register_primitive_function, call_primitive_func
from hidet.ir.type import FuncType
from hidet.utils import initialize


@initialize()
def register_primitive_functions():
    functions = [
        ('cuda_activemask', '__activemask', FuncType([], 'int32')),
        # T __shfl_sync(unsigned mask, T var, int srcLane, int width=warpSize)
        ('cuda_shfl_sync', '__shfl_sync', FuncType(type_infer_func=lambda arg_types: arg_types[1])),
        ('cuda_shfl_up_sync', '__shfl_up_sync', FuncType(type_infer_func=lambda arg_types: arg_types[1])),
        ('cuda_shfl_down_sync', '__shfl_down_sync', FuncType(type_infer_func=lambda arg_types: arg_types[1])),
    ]
    for name, codegen_name, func_type in functions:
        register_primitive_function(name=name, func_or_type=func_type, codegen_name=codegen_name)


def shfl_sync(mask, var, src_lane, width=32):
    return call_primitive_func('cuda_shfl_sync', [mask, var, src_lane, width])


def shfl_up_sync(mask, var, delta, width=32):
    return call_primitive_func('cuda_shfl_up_sync', [mask, var, delta, width])


def shfl_down_sync(mask, var, delta, width=32):
    return call_primitive_func('cuda_shfl_down_sync', [mask, var, delta, width])


def shfl_xor_sync(mask, var, lane_mask, width=32):
    return call_primitive_func('cuda_shfl_down_sync', [mask, var, lane_mask, width])


def active_mask():
    return call_primitive_func('cuda_activemask', [])
