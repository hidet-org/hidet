def concat_op_name(lhs: str, rhs: str) -> str:
    # lhs = lhs[5:] if lhs.startswith('Fused') else lhs
    # rhs = rhs[5:] if rhs.startswith('Fused') else rhs
    # return 'Fused{}{}'.format(lhs, rhs)
    return '{} {}'.format(lhs, rhs)
