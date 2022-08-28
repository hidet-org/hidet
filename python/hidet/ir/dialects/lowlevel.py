from __future__ import annotations
from typing import Optional, Union, Sequence, List
from hidet.ir.type import TypeNode, ScalarType, TensorType, Scope, Int, tensor_type
from hidet.ir.expr import Expr, TensorElement, Var, Constant, cast
from hidet.ir.layout import DataLayout


class VoidType(TypeNode):
    pass


class PointerType(TypeNode):
    def __init__(self, base_type, specifiers: Optional[Sequence[str]] = None, use_bracket: bool = False):
        super().__init__()
        if isinstance(base_type, str):
            base_type = ScalarType(base_type)
        self.base_type: TypeNode = base_type
        self.specifiers: List[str] = list(specifiers) if specifiers else []
        self.use_bracket: bool = use_bracket


class ReferenceType(TypeNode):
    def __init__(self, base_type):
        super().__init__()
        self.base_type = base_type


class TensorPointerType(TypeNode):
    def __init__(self,
                 scope: Optional[Union[Scope, str]] = None,
                 dtype: Optional[Union[ScalarType, str]] = None,
                 shape: Optional[Sequence[Int]] = None,
                 layout: Optional[Union[Sequence[Int], DataLayout]] = None):
        self.tensor_type: TensorType = tensor_type(scope, dtype, shape, layout)

    @staticmethod
    def from_tensor_type(tp: TensorType) -> TensorPointerType:
        tpt = object.__new__(TensorPointerType)
        tpt.tensor_type = tp
        return tpt


#
# Moved to hidet.ir.expr
#
# class Cast(Expr):
#     def __init__(self, expr, target_type):
#         self.expr = expr
#         if isinstance(target_type, str):
#             target_type = ScalarType(target_type)
#         self.target_type = target_type


class Dereference(Expr):
    def __init__(self, expr):
        self.expr = expr


class Address(Expr):
    def __init__(self, expr):
        self.expr = expr


class Reference(Expr):
    def __init__(self, expr):
        assert isinstance(expr, (TensorElement, Var)), "only l-value can be referenced."
        self.expr = expr


def pointer_type(base_type):
    return PointerType(base_type)


def tensor_pointer_var(hint: str, shape=None, scope: str = 'global', dtype: Union[str, ScalarType] = 'float32', layout=None):
    return Var(hint, TensorPointerType(scope=scope, dtype=dtype, shape=shape, layout=layout))


def void_pointer():
    return PointerType(VoidType())


def view(ptr: Expr, tp: TensorType) -> Expr:
    if not isinstance(tp, TensorType):
        raise ValueError('Expect a tensor type, got {}'.format(type(tp).__name__))
    return cast(ptr, TensorPointerType.from_tensor_type(tp))


void_p = PointerType(VoidType())
