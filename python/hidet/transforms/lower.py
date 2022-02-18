from hidet.ir.func import IRModule
from hidet.transforms import *


def lower(ir_module: IRModule) -> IRModule:
    transforms = [
        # eliminate_dead_device_function_pass(),
        generate_packed_func_pass(),
        flatten_tensor_pass(),

        expression_simplification_pass(),
        # build_let_stmt_pass(),

        simplify_stmt_pass(),
        # expand_let_expr_pass(),
        # bound_aware_simplify_pass(),
        # eliminate_dead_let_stmt_pass(),
        # common_subexpression_elimination_pass(),
        # vectorize_load_store_pass(),     # disable by default, this optimization can be conducted automatically by underlying ptxas assembler.
        import_primitive_functions_pass()
    ]

    for transform in transforms:
        ir_module = transform(ir_module)

    return ir_module

