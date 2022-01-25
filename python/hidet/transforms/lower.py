from hidet.ir.func import IRModule
from hidet.transforms import *


def lower(ir_module: IRModule) -> IRModule:
    transforms = [
        eliminate_dead_device_function_pass(),
        generate_packed_func_pass(),
        flatten_tensor_pass(),
        const_expr_simplifier_pass(),
        bound_aware_simplify_pass(),
        # vectorize_load_store_pass(),
        import_primitive_functions_pass()
    ]

    for transform in transforms:
        ir_module = transform(ir_module)

    return ir_module

