from .bound_aware_simplify import bound_aware_simplify_pass
from .inline_let_stmt import inline_let_stmt_pass
from .expand_let_expr import expand_let_expr_pass
from .common_subexpression_elimination import common_subexpression_elimination_pass
from .take_out_constant import take_out_constant_pass
from .build_let_stmt import build_let_stmt_pass
from ..base import RepeatFunctionPass, SequencePass


def expression_simplification_pass():
    return SequencePass([
        expand_let_expr_pass(),
        inline_let_stmt_pass(inline_all=True),
        take_out_constant_pass(),
        bound_aware_simplify_pass(),
        # build_let_stmt_pass(),
        common_subexpression_elimination_pass(),
        inline_let_stmt_pass()
    ])
    # return RepeatFunctionPass(
    #     name='ExpressionSimplificationPass',
    #     passes=[
    #         expand_let_expr_pass(),
    #         inline_let_stmt_pass(),
    #         bound_aware_simplify_pass(),
    #         inline_let_stmt_pass(),
    #         take_out_constant_pass(),
    #         common_subexpression_elimination_pass(),
    #     ],
    #     repeat_limit=10
    # )
