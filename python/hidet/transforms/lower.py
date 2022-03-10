from hidet.ir.func import IRModule
from hidet.transforms import *


def lower(ir_module: IRModule) -> IRModule:
    transforms = [
        # necessary passes
        # eliminate_dead_device_function_pass(),
        generate_packed_func_pass(),
        flatten_tensor_pass(),
        expand_let_expr_pass(),


        # simplification
        # explicit_unroll_for_stmt_pass(),
        inline_let_stmt_pass(inline_all=True),
        rule_based_simplify_pass(),
        simplify_stmt_pass(),

        # common sub-expression elimination
        build_let_stmt_pass(),
        uplift_let_stmt_pass(),
        common_subexpression_elimination_pass(),
        inline_let_stmt_pass(inline_factor=1),

        # necessary pass
        import_primitive_functions_pass()
    ]

    for transform in transforms:
        ir_module = transform(ir_module)

    return ir_module

