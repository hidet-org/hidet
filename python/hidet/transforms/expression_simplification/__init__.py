from .bound_aware_simplify import bound_aware_simplify_pass
from .inline_let_stmt import inline_let_stmt_pass
from .expand_let_expr import expand_let_expr_pass
from .common_subexpression_elimination import common_subexpression_elimination_pass
from .take_out_constant import take_out_constant_pass
from .build_let_stmt import build_let_stmt_pass
from .rule_based_simplifier import rule_based_simplify_pass
from ..base import RepeatFunctionPass, SequencePass


def expression_simplification_pass():
    return SequencePass([
        expand_let_expr_pass(),
        inline_let_stmt_pass(inline_all=True),
        # rule_based_simplify_pass(),
        take_out_constant_pass(),
        # build_let_stmt_pass(),
        # common_subexpression_elimination_pass(),
        # inline_let_stmt_pass(inline_factor=2)
    ])
