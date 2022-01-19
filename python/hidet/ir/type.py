from typing import Sequence, Optional, Union, Dict, Tuple, List, Callable
from hidet.ir.node import Node


class BaseType(Node):
    pass


class Scope:
    def __init__(self, name):
        assert name in ['host', 'global', 'shared', 'register']
        self.name = name


class RegisterScope(Scope):
    def __init__(self, global2local=None, local2global=None, local_shape=None):
        from hidet.ir.expr import Expr
        super().__init__('register')
        self.global2local: Callable[[Expr, ...], List[Expr, ...]] = global2local
        self.local2global: Callable[[Expr, Expr, ...], List[Expr, ...]] = local2global
        self.local_shape: Tuple[int, ...] = local_shape


class ScalarType(BaseType):
    def __init__(self, name):
        if name:
            assert name in ['float32', 'int32', 'bool'], name
        self.name = name


class TensorType(BaseType):
    def __init__(self, scope: Optional[Union[Scope, str]], scalar_type: Union[ScalarType, str], shape=None, strides=None):
        from hidet.ir.expr import convert
        if isinstance(scope, str):
            scope = Scope(scope)
        if isinstance(scalar_type, str):
            scalar_type = ScalarType(scalar_type)
        if strides:
            strides = [convert(s) for s in strides]
        self.scope: Scope = scope
        self.scalar_type: ScalarType = scalar_type
        self.shape = shape
        self.strides = strides

    def nbytes(self):
        from hidet.ir.expr import convert, Constant
        from hidet.ir.functors import simplify
        max_index_expr = sum([(a - 1) * b for a, b in zip(self.shape, self.strides)], convert(0))
        max_index_value = simplify(max_index_expr)
        if isinstance(max_index_value, Constant):
            return max_index_value.value
        else:
            raise Exception("Can only calculate size of static tensor.")


class FuncType(BaseType):
    def __init__(self, param_types, ret_type):
        self.param_types = param_types
        self.ret_type = ret_type

    @staticmethod
    def from_func(func):
        return FuncType([param.type for param in func.params], func.ret_type)


def scalar_type(type_name):
    return ScalarType(type_name)


def tensor_type(scope, scalar_type, shape, strides=None):
    if isinstance(scope, str):
        scope = Scope(scope)
    from hidet.ir.expr import convert
    if isinstance(scalar_type, str):
        scalar_type = ScalarType(scalar_type)
    assert isinstance(scalar_type, ScalarType)
    shape = [convert(s) for s in shape]
    if strides is None:
        # use default compact strides, row-major
        strides = []
        n = len(shape)
        for i in range(n):
            c = convert(1)
            for j in range(i + 1, n):
                c = c * shape[j]
            strides.append(c)
    strides = [convert(s) for s in strides]
    return TensorType(scope, scalar_type, shape, strides)


