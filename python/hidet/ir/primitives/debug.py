from hidet.ir.stmt import BlackBoxStmt


def printf(format_string, *args):
    """
    usage:
    printf("%d %d\n", expr_1, expr_2)
    """
    format_string = format_string.replace('\n', '\\n')
    if len(args) > 0:
        arg_string = ', '.join(['{}'] * len(args))
        template_string = f'printf("{format_string}", {arg_string});'
    else:
        template_string = f'printf("{format_string}");'
    # if '\n' in format_string:
    #     raise ValueError('Please use printf(r"...\\n") instead of printf("...\\n").')
    return BlackBoxStmt(template_string, *args)
