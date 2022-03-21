from hidet.ir.func import IRModule
from hidet.transforms import *


def lower(ir_module: IRModule) -> IRModule:
    transforms = [
        # necessary passes
        generate_packed_func_pass(),
        flatten_tensor_pass(),
        expand_let_expr_pass(),

        # simplification
        inline_let_stmt_pass(inline_all=True),
        rule_based_simplify_pass(),
        simplify_stmt_pass(),

        # common sub-expression elimination
        build_let_stmt_pass(),
        uplift_let_stmt_pass(),
        common_subexpression_elimination_pass(),
        # inline_let_stmt_pass(inline_factor=1),
        inline_let_stmt_pass(inline_all=True),

        # optimization (precompute condition)
        precompute_condition_pass(),


        # necessary pass
        import_primitive_functions_pass()
    ]

    for transform in transforms:
        ir_module = transform(ir_module)

    return ir_module

