from typing import Union, Optional, Sequence
from hidet.ir.type import TypeNode, DataType, tensor_type
from hidet.ir.layout import DataLayout
from hidet.ir.expr import Expr, cast
from hidet.ir.stmt import DeclareScope


class TypeDecorator:
    def __init__(self, decorated_type: TypeNode, scope: Optional[Union[str, DeclareScope]], is_static: bool = False):
        if isinstance(scope, str):
            scope = DeclareScope.from_str(scope)
        self.decorated_type: TypeNode = decorated_type
        self.scope = scope
        self.is_static = is_static


def static(tp: Union[TypeNode, TypeDecorator]):
    if isinstance(tp, TypeNode):
        return TypeDecorator(decorated_type=tp, scope=None, is_static=True)
    else:
        return TypeDecorator(decorated_type=tp.decorated_type, scope=tp.scope, is_static=True)


def with_scope(scope: Union[str, DeclareScope], tp: Union[TypeNode, TypeDecorator]):
    if isinstance(tp, TypeNode):
        return TypeDecorator(decorated_type=tp, scope=scope, is_static=False)
    else:
        return TypeDecorator(decorated_type=tp.decorated_type, scope=scope, is_static=tp.is_static)


def shared_scope(tp: Union[TypeNode, TypeDecorator]):
    return with_scope(DeclareScope.Shared, tp)


def register_scope(tp: Union[TypeNode, TypeDecorator]):
    return with_scope(DeclareScope.Register, tp)


def global_scope(tp: Union[TypeNode, TypeDecorator]):
    return with_scope(DeclareScope.Global, tp)


def tensor(
    scope: Union[DeclareScope, str],
    dtype: Union[DataType, str],
    shape: Optional[Sequence[Union[Expr, int]]] = None,
    layout: Optional[DataLayout] = None,
):
    return with_scope(scope, tensor_type(dtype, shape, layout))


def tensor_pointer(
    dtype: Union[DataType, str], shape: Optional[Sequence[Union[Expr, int]]] = None, layout: Optional[DataLayout] = None
):
    return ~tensor_type(dtype, shape, layout)


def as_tensor_pointer(
    expr: Expr,
    dtype: Union[DataType, str],
    shape: Optional[Sequence[Union[Expr, int]]] = None,
    layout: Optional[DataLayout] = None,
) -> Expr:
    return cast(expr, tensor_pointer(dtype, shape, layout))


def shared_tensor(
    dtype: Union[DataType, str], shape: Optional[Sequence[Union[Expr, int]]] = None, layout: Optional[DataLayout] = None
):
    return shared_scope(tensor_type(dtype, shape, layout))


def register_tensor(
    dtype: Union[DataType, str], shape: Optional[Sequence[Union[Expr, int]]] = None, layout: Optional[DataLayout] = None
):
    return register_scope(tensor_type(dtype, shape, layout))
