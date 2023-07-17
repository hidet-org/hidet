from hidet.ir.tools import infer_type
from hidet.ir.type import TensorType, TensorPointerType
from hidet.ir.expr import Expr


def get_tensor_type(expr: Expr) -> TensorType:
    expr_type = infer_type(expr)
    if isinstance(expr_type, TensorType):
        return expr_type
    elif isinstance(expr_type, TensorPointerType):
        return expr_type.tensor_type
    else:
        raise TypeError('Can not infer the expr type to get a tensor type, got {}'.format(expr_type))
