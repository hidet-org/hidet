# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# pylint: disable=import-outside-toplevel, useless-parent-delegation, redefined-outer-name, redefined-builtin
# pylint: disable=useless-super-delegation
from typing import Optional, Union, Sequence, Tuple
import string
import numpy as np
from .node import Node
from .type import TypeNode, TensorType, DataType, TensorPointerType, PointerType, FuncType, tensor_type, data_type

PyScalar = Union[int, float]


class Expr(Node):
    def __neg__(self):
        return Neg(self)

    def __add__(self, other):
        return Add(self, other)

    def __radd__(self, other):
        return Add(other, self)

    def __sub__(self, other):
        return Sub(self, other)

    def __rsub__(self, other):
        return Sub(other, self)

    def __mul__(self, other):
        return Multiply(self, other)

    def __rmul__(self, other):
        return Multiply(other, self)

    def __truediv__(self, other):
        return Div(self, other)

    def __rtruediv__(self, other):
        return Div(other, self)

    def __floordiv__(self, other):
        return Div(self, other)

    def __rfloordiv__(self, other):
        return Div(other, self)

    def __mod__(self, other):
        return Mod(self, other)

    def __rmod__(self, other):
        return Mod(other, self)

    def __lt__(self, other):
        return LessThan(self, other)

    def __le__(self, other):
        return LessEqual(self, other)

    def __eq__(self, other):
        return Equal(self, other)

    def __hash__(self):
        return id(self)

    def __gt__(self, other):
        return LessThan(other, self)

    def __ge__(self, other):
        return LessEqual(other, self)

    def __invert__(self):
        """
        We override the invert operator ~a as the addressing operator.
        """
        return Address(self)

    def __or__(self, other):
        return BitwiseOr(self, other)

    def __ror__(self, other):
        return BitwiseOr(other, self)

    def __and__(self, other):
        return BitwiseAnd(self, other)

    def __rand__(self, other):
        return BitwiseAnd(other, self)

    def __xor__(self, other):
        return BitwiseXor(self, other)

    def __rxor__(self, other):
        return BitwiseXor(other, self)

    def __getitem__(self, items):
        if not isinstance(items, (tuple, list)):
            items = [items]
        indices = []
        starts = []
        ends = []
        for item in items:
            if isinstance(item, slice):
                indices.append(None)
                starts.append(item.start)
                ends.append(item.stop)
                assert item.step is None, "do not support step slice"
            else:
                indices.append(item)
                starts.append(None)
                ends.append(None)
        rank = tensor_rank(self)
        if len(items) < rank or any(i is None for i in indices):
            while len(indices) < rank:
                indices.append(None)
                starts.append(None)
                ends.append(None)
            return TensorSlice(base=self, indices=indices, starts=starts, ends=ends)
        else:
            return TensorElement(base=self, indices=indices)

    def __setitem__(self, key, value):
        raise ValueError()

    def __int__(self):
        assert isinstance(self, Constant), 'Expect a Constant, got {} with type {}'.format(self, type(self))
        return int(self)

    def __float__(self):
        assert isinstance(self, Constant), 'Expect a Constant, got {} with type {}'.format(self, type(self))
        return float(self)

    def __str__(self):
        from hidet.ir.functors import astext

        return str(astext(self))

    def equals(self, other):
        return Equal(self, other)

    def is_const(self):
        return isinstance(self, Constant)

    def const(self) -> 'Constant':
        assert isinstance(self, Constant)
        return self

    def read(self, items, protected=True):
        te = self[items]
        if not isinstance(te, TensorElement):
            raise ValueError('expect element indexing, but got slicing.')
        te.protected = protected
        return te

    def write(self, items, value, protected=True):
        from hidet.ir.stmt import BufferStoreStmt

        te = self[items]
        if not isinstance(te, TensorElement):
            raise ValueError('expect element indexing, but got slicing.')
        return BufferStoreStmt(self, te.indices, value, protected)


# the following are used as type hints
# if a primitive function expect an int8 expression, we should use ExprInt8 instead of Expr
# to explicit tell the reader that this function expect an int8 expression
# but this is not enforced by the type system of python
# ExprInt8 should be read as "expression with int8 type"
# Usage example:
#
#   def cuda_i64_to_f16(a: ExprInt64) -> ExprFloat16:
#       ...
# Above function expects an int64 expression and returns a float16 value.

ExprInt8 = ExprInt16 = ExprInt32 = ExprInt64 = Expr
ExprUInt8 = ExprUInt16 = ExprUInt32 = ExprUInt64 = Expr
ExprFloat16 = ExprFloat32 = ExprFloat64 = ExprBFloat16 = ExprTFloat32 = Expr


class BinaryOp(Expr):
    def __init__(self, a, b):
        self.a = convert(a)
        self.b = convert(b)


class UnaryOp(Expr):
    def __init__(self, a):
        self.a = convert(a)


def convert(
    obj: Optional[Union[Expr, PyScalar, tuple, Sequence]], dtype: Optional[Union[str, DataType]] = None
) -> Optional[Union[Expr, tuple]]:
    if isinstance(obj, Expr):
        return obj

    if dtype is not None:
        if isinstance(obj, (bool, int, float)):
            return Constant(obj, dtype)
        else:
            raise ValueError('Can not convert {} to {}.'.format(obj, dtype))

    if isinstance(obj, bool):
        return Constant(obj, data_type('bool'))
    elif isinstance(obj, int):
        return Constant(obj, data_type('int32'))
    elif isinstance(obj, float):
        return Constant(obj, data_type('float32'))
    elif isinstance(obj, (tuple, list)):
        return tuple(convert(v) for v in obj)
    elif obj is None:
        return None
    else:
        raise NotImplementedError(type(obj))


class Condition(Expr):
    pass


class LessThan(Condition, BinaryOp):
    def __init__(self, a, b):
        super().__init__(a, b)


class LessEqual(Condition, BinaryOp):
    def __init__(self, a, b):
        super().__init__(a, b)


class Equal(Condition, BinaryOp):
    def __init__(self, a, b):
        super().__init__(a, b)

    def __bool__(self):
        r = object.__eq__(self.a, self.b)
        if r is NotImplemented:
            return False
        else:
            return True


class NotEqual(Condition, BinaryOp):
    def __init__(self, a, b):
        super().__init__(a, b)


class LogicalAnd(Condition, BinaryOp):
    def __init__(self, a, b):
        super().__init__(a, b)

    @staticmethod
    def join(*conds):
        cond = None
        for c in conds:
            cond = LogicalAnd(cond, convert(c)) if cond is not None else convert(c)
        return cond

    @staticmethod
    def join_list(conds: Sequence[Condition]):
        return LogicalAnd.join(*conds)


class LogicalOr(Condition, BinaryOp):
    def __init__(self, a, b):
        super().__init__(a, b)

    @staticmethod
    def join(*conds):
        cond = None
        for c in conds:
            cond = LogicalOr(cond, convert(c)) if cond is not None else convert(c)
        return cond

    @staticmethod
    def join_list(conds: Sequence[Condition]):
        return LogicalOr.join(*conds)


class LogicalNot(Condition, UnaryOp):
    def __init__(self, a):
        super().__init__(a)


class Neg(UnaryOp):
    def __init__(self, a):
        super().__init__(a)


class Add(BinaryOp):
    def __init__(self, a, b):
        super().__init__(a, b)


class Sub(BinaryOp):
    def __init__(self, a, b):
        super().__init__(a, b)


class Multiply(BinaryOp):
    def __init__(self, a, b):
        super().__init__(a, b)


class Div(BinaryOp):
    def __init__(self, a, b):
        super().__init__(a, b)


class FloorDiv(BinaryOp):
    def __init__(self, a, b):
        super().__init__(a, b)


class Mod(BinaryOp):
    def __init__(self, a, b):
        super().__init__(a, b)


class BitwiseNot(Expr):
    def __init__(self, base):
        super().__init__()
        self.base = base


class BitwiseAnd(BinaryOp):
    def __init__(self, a, b):
        super().__init__(a, b)


class BitwiseOr(BinaryOp):
    def __init__(self, a, b):
        super().__init__(a, b)

    @staticmethod
    def join_list(lst):
        if len(lst) == 0:
            return convert(0)
        else:
            current = lst[0]
            for v in lst[1:]:
                current = BitwiseOr(current, v)
            return current


class BitwiseXor(BinaryOp):
    def __init__(self, a, b):
        super().__init__(a, b)


class LeftShift(Expr):
    def __init__(self, base, cnt):
        super().__init__()
        self.base = convert(base)
        self.cnt = convert(cnt)


class RightShift(Expr):
    def __init__(self, base, cnt):
        super().__init__()
        self.base = base
        self.cnt = cnt


class TensorElement(Expr):
    def __init__(self, base, indices, protected=False):
        self.base = base
        self.indices = convert(indices)
        self.protected: bool = protected


class TensorSlice(Expr):
    def __init__(self, base, indices, starts, ends):
        # a[3, 4:, :5, :] will be represented by
        # base: a
        # indices: [3, None, None, None]
        # starts: [None, 4, None, None]
        # ends: [None, None, 5, None]
        self.base = base
        self.indices: Tuple = convert(indices)
        self.starts: Tuple = convert(starts)
        self.ends: Tuple = convert(ends)
        if self.base is not None:
            assert len(self.indices) == tensor_rank(base)


class Call(Expr):
    def __init__(self, func_var, args):
        self.func_var: Var = func_var
        self.args = convert(args)


class Let(Expr):
    def __init__(self, var, value, body):
        self.var = var
        self.value = convert(value)
        self.body = convert(body)


class Cast(Expr):
    def __init__(self, expr, target_type: TypeNode):
        assert isinstance(target_type, TypeNode)
        self.expr = expr
        self.target_type: TypeNode = target_type


class Constant(Expr):
    def __init__(self, value=None, const_type=None):
        if const_type and isinstance(const_type, str):
            const_type = data_type(const_type)
        self.value: Optional[np.ndarray, float, int] = value
        self.type: Optional[Union[DataType, TensorType]] = const_type

    def is_scalar(self) -> bool:
        return self.type and isinstance(self.type, DataType)

    def is_tensor(self) -> bool:
        return self.type and isinstance(self.type, TensorType)

    def __int__(self):
        return int(self.value)

    def __float__(self):
        return float(self.value)

    def array(self) -> np.ndarray:
        return self.value


class IfThenElse(Expr):
    """
    The if-then-else expression.

    Parameters
    ----------
    cond: Expr
        The condition of the if-then-else expression.
    then_expr: Expr
        The expression to be evaluated if the condition is true.
    else_expr: Expr
        The expression to be evaluated if the condition is false.
    """

    def __init__(self, cond: Union[Expr, PyScalar], then_expr: Union[Expr, PyScalar], else_expr: Union[Expr, PyScalar]):
        self.cond = convert(cond)
        self.then_expr = convert(then_expr)
        self.else_expr = convert(else_expr)


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


class Var(Expr):
    id_clock = 0

    def __init__(self, hint: Optional[str], type: TypeNode, name: Optional[str] = None):
        """
        A variable may have a hint, name, and id.

        Hint is used to determine the name in codegen. Different vars may have the
        same hint. If two vars have the same hint such as 'x', the final name would be like 'x1', 'x2'.

        OUTDATED:
        Name is the determined name in the final code. Used by primitive varaibles such as 'threadIdx.x'. No variable
        should have a same name as primitive objects (including primitive variables and primitive functions).

        Id is used to track the allocation of Var object in python, which is only used to help us to distinguish
        different Var in python debugger.
        """
        self.hint = hint
        self.name = name
        self.type: Union[TypeNode, TensorType, TensorPointerType, FuncType] = type
        self.id = self.new_id()

    @staticmethod
    def new_id():
        Var.id_clock += 1
        return Var.id_clock

    @staticmethod
    def reset_id_counter():
        Var.id_clock = 0


def var(hint: str = None, dtype='int32'):
    if isinstance(hint, str):
        assert set(hint) <= set(string.ascii_letters + '_.' + string.digits)
    return Var(hint, data_type(dtype))


def scalar_var(hint: str, dtype: Union[str, DataType] = 'float32') -> Var:
    dtype = dtype if isinstance(dtype, DataType) else data_type(dtype)
    return Var(hint, dtype)


def tensor_var(hint: str, shape=None, dtype: Union[str, DataType] = 'float32', layout=None) -> Var:
    return Var(hint, tensor_type(dtype, shape, layout))


def is_one(v: Expr) -> bool:
    return isinstance(v, Constant) and v.value == 1


def is_zero(v: Expr) -> bool:
    return isinstance(v, Constant) and v.value == 0


def is_true(v: Expr) -> bool:
    return isinstance(v, Constant) and v.type.name == 'bool' and v.value is True


def is_false(v: Expr) -> bool:
    return isinstance(v, Constant) and v.type.name == 'bool' and v.value is False


def is_const_int(v: Expr) -> bool:
    return isinstance(v, Constant) and v.type.name == 'int32'


def if_then_else(
    cond: Union[Expr, PyScalar], then_expr: Union[Expr, PyScalar], else_expr: Union[Expr, PyScalar]
) -> IfThenElse:
    """
    Create an if-then-else expression.

    Parameters
    ----------
    cond: Expr or PyScalar
        The condition of the if-then-else expression.

    then_expr: Expr or PyScalar
        The expression to be evaluated if the condition is true.

    else_expr: Expr or PyScalar
        The expression to be evaluated if the condition is false.

    Returns
    -------
    ret: IfThenElse
        The if-then-else expression.
    """
    return IfThenElse(convert(cond), convert(then_expr), convert(else_expr))


def is_tensor(v: Expr) -> bool:
    if not isinstance(v, Var):
        return False
    return isinstance(v.type, (TensorType, TensorPointerType))


def get_tensor_layout(v: Expr):
    assert isinstance(v, Var) and isinstance(v.type, (TensorType, TensorPointerType))
    return v.type.layout if isinstance(v.type, TensorType) else v.type.tensor_type.layout


def tensor_rank(v: Expr) -> int:
    from hidet.ir.compute import TensorNode

    if isinstance(v, Var):
        if isinstance(v.type, TensorType):
            return len(v.type.shape)
        elif isinstance(v.type, TensorPointerType):
            return len(v.type.tensor_type.shape)
        elif isinstance(v.type, PointerType):
            return 1
        else:
            raise ValueError(v)
    elif isinstance(v, TensorSlice):
        return sum(1 if i is None else 0 for i in v.indices)
    elif isinstance(v, TensorNode):
        return len(v.ttype.shape)
    elif isinstance(v, Constant) and isinstance(v.type, TensorType):
        return len(v.type.shape)
    elif isinstance(v, Cast) and isinstance(v.target_type, PointerType):
        return 1
    else:
        raise ValueError('Can not infer the tensor rank of "{}"'.format(v))


def cast(v: Expr, dtype: Union[str, DataType, TypeNode]):
    if isinstance(dtype, str):
        dtype = data_type(dtype)
    return Cast(v, dtype)


def const_tensor(value: np.ndarray) -> Constant:
    from hidet.ir.utils.type_utils import from_numpy_dtype

    return Constant(value=value, const_type=tensor_type(dtype=from_numpy_dtype(value.dtype), shape=list(value.shape)))


def tensor_pointer_var(hint: str, shape=None, dtype: Union[str, DataType] = 'float32', layout=None):
    return Var(hint, TensorPointerType(dtype=dtype, shape=shape, layout=layout))


def view(ptr: Expr, tp: TensorType) -> Expr:
    if not isinstance(tp, TensorType):
        raise ValueError('Expect a tensor type, got {}'.format(type(tp).__name__))
    return cast(ptr, TensorPointerType.from_tensor_type(tp))


def address(v: Expr) -> Expr:
    return Address(v)


def deref(v: Expr) -> Expr:
    return Dereference(v)
