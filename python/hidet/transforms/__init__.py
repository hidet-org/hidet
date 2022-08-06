from hidet.ir.func import IRModule

from .base import Pass, FunctionPass, FunctionBodyPass, SequencePass, RepeatFunctionPass, PassContext
from .instruments import PassInstrument, SaveIRInstrument, ProfileInstrument

from .apply_prologue_epilogue import apply_prologue_epilogue_pass
from .flatten_tensor_slice import flatten_tensor_slice_pass
from .flatten_tensor_index import flatten_tensor_index_pass
from .generate_packed_func import generate_packed_func_pass
from .import_primitive_functions import import_primitive_functions_pass
from .simplify_stmt import simplify_stmt_pass
from .expand_let_expr import expand_let_expr_pass
from .resolve_generic_primitive_function import resolve_primitive_func_pass
from .add_explicit_cast import add_explicit_cast_pass
from .explicit_unroll_for_stmt import explicit_unroll_for_stmt_pass
from .inline_let_stmt import inline_let_stmt_pass
from .common_subexpression_elimination import common_subexpression_elimination_pass, chain_seq_stmt_using_let_stmt_pass
from .build_let_stmt import build_let_stmt_pass
from .rule_based_simplifier import rule_based_simplify_pass
from .simplify_stmt import simplify_stmt_pass
from .squeeze_let_stmt import squeeze_let_stmt_pass
from .uplift_let_stmt import uplift_let_stmt_pass
from .precompute_condition import precompute_condition_pass
from .normalize_const_tensor import normalize_const_tensor_pass
from .lower_task_mapping import lower_task_mapping_pass


def lower(ir_module: IRModule) -> IRModule:
    transforms = [
        # necessary passes
        flatten_tensor_slice_pass(),
        lower_task_mapping_pass(),
        apply_prologue_epilogue_pass(),
        generate_packed_func_pass(),
        normalize_const_tensor_pass(),
        rule_based_simplify_pass(),  # make ir more readable
        flatten_tensor_index_pass(),
        resolve_primitive_func_pass(),
        import_primitive_functions_pass(),
        resolve_primitive_func_pass(),
        import_primitive_functions_pass(),
        add_explicit_cast_pass(),

        # simplification
        expand_let_expr_pass(),
        inline_let_stmt_pass(inline_all=True),
        rule_based_simplify_pass(),
        simplify_stmt_pass(),

        # common sub-expression elimination
        # build_let_stmt_pass(),
        # uplift_let_stmt_pass(),
        # common_subexpression_elimination_pass(),
        # inline_let_stmt_pass(inline_factor=1),

        # optimization (precompute condition)
        # precompute_condition_pass(),

        # necessary pass
    ]

    ctx = PassContext.current()
    for instrument in ctx.instruments:
        instrument.before_all_passes(ir_module)
    for transform in transforms:
        ir_module = transform(ir_module)
    for instrument in ctx.instruments:
        instrument.after_all_passes(ir_module)

    return ir_module
