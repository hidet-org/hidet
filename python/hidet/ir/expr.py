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
from __future__ import annotations
from typing import Optional, Union, Sequence, Tuple, Dict, Type, Callable
import string
import operator
import numpy as np
from .node import Node
from .type import TypeNode, TensorType, DataType, TensorPointerType, PointerType, FuncType, tensor_type, data_type
from .type import tensor_pointer_type

PyScalar = Union[bool, int, float, complex]


class Expr(Node):
    def __bool__(self):
        raise TypeError(
            "hidet.ir.Expr does not support pythonic logical operations (e.g., and, or, not, if(...)). "
            "Please use hidet.ir.if_then_else, hidet.ir.LogicalAnd, hidet.ir.LogicalOr, hidet.ir.LogicalNot explicitly."
        )

    def __call__(self, *args, **kwargs):
        if len(kwargs) > 0:
            raise ValueError("Keyword arguments are not supported in hidet function calls.")
        if isinstance(self, Var) and self.type.is_func_type():
            return call(self, args)
        else:
            raise ValueError("Only function variable can be called.")

    def __neg__(self):
        return self._unary(Neg, self)

    def __add__(self, other):
        return self._binary(Add, self, other)

    def __radd__(self, other):
        return self._binary(Add, other, self)

    def __sub__(self, other):
        return self._binary(Sub, self, other)

    def __rsub__(self, other):
        return self._binary(Sub, other, self)

    def __mul__(self, other):
        return self._binary(Multiply, self, other)

    def __rmul__(self, other):
        return self._binary(Multiply, other, self)

    def __truediv__(self, other):
        return self._binary(Div, self, other)

    def __rtruediv__(self, other):
        return self._binary(Div, other, self)

    def __floordiv__(self, other):
        return self._binary(Div, self, other)

    def __rfloordiv__(self, other):
        return self._binary(Div, other, self)

    def __mod__(self, other):
        return self._binary(Mod, self, other)

    def __rmod__(self, other):
        return self._binary(Mod, other, self)

    def __lt__(self, other):
        return self._binary(LessThan, self, other)

    def __le__(self, other):
        return self._binary(LessEqual, self, other)

    def __eq__(self, other):
        return self._binary(Equal, self, other)

    def __ne__(self, other):
        return self._binary(NotEqual, self, other)

    def __lshift__(self, other):
        return self._binary(LeftShift, self, other)

    def __rshift__(self, other):
        return self._binary(RightShift, self, other)

    def __hash__(self):
        return id(self)

    def __gt__(self, other):
        return self._binary(LessThan, other, self)

    def __ge__(self, other):
        return self._binary(LessEqual, other, self)

    def __invert__(self):
        return Address(self)  # We override the invert operator `~a` as the addressing operator.

    def __or__(self, other):
        return self._binary(BitwiseOr, self, other)

    def __ror__(self, other):
        return self._binary(BitwiseOr, other, self)

    def __and__(self, other):
        return self._binary(BitwiseAnd, self, other)

    def __rand__(self, other):
        return self._binary(BitwiseAnd, other, self)

    def __xor__(self, other):
        return self._binary(BitwiseXor, self, other)

    def __rxor__(self, other):
        return self._binary(BitwiseXor, other, self)

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
            return tensor_slice(base=self, indices=indices, starts=starts, ends=ends)
        else:
            return tensor_element(base=self, indices=indices)

    def __setitem__(self, key, value):
        raise ValueError()

    def __str__(self):
        from hidet.ir.tools import astext

        return str(astext(self))

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

    @staticmethod
    def _unary(cls, a):  # pylint: disable=bad-staticmethod-argument
        if not isinstance(a, Expr):
            a = convert(a)
        if isinstance(a, Constant):
            if cls is Neg:
                return Constant(-a.value, a.type)
            elif cls is LogicalNot:
                return Constant(not a.value, a.type)
            elif cls is BitwiseNot:
                return Constant(~a.value, a.type)
            else:
                raise ValueError('unknown unary operator {}'.format(cls))
        else:
            return cls(a)

    @staticmethod
    def _binary(cls, a, b):  # pylint: disable=bad-staticmethod-argument
        if not isinstance(a, Expr):
            a = convert(a)
        if not isinstance(b, Expr):
            b = convert(b)
        if isinstance(a, Constant) and isinstance(b, Constant):
            from hidet.ir.dtypes import promote_type

            value = _operator_dict[cls](a.value, b.value)
            if cls in [Equal, NotEqual, LessThan, LessEqual, LogicalAnd, LogicalOr]:
                return Constant(value, const_type='bool')
            elif cls in [LeftShift, RightShift]:
                return Constant(value, a.type)
            else:
                return Constant(value, promote_type(a.type, b.type))
        else:
            return cls(a, b)


class BinaryExpr(Expr):
    def __init__(self, a: Expr, b: Expr):
        self.a: Expr = a
        self.b: Expr = b

        assert isinstance(a, Expr)
        assert isinstance(b, Expr)


class UnaryExpr(Expr):
    def __init__(self, a: Expr):
        self.a: Expr = convert(a)

        assert isinstance(a, Expr)


class Condition(Expr):
    pass


class LessThan(Condition, BinaryExpr):
    def __init__(self, a, b):
        super().__init__(a, b)


class LessEqual(Condition, BinaryExpr):
    def __init__(self, a, b):
        super().__init__(a, b)


class Equal(Condition, BinaryExpr):
    def __init__(self, a, b):
        super().__init__(a, b)

    def __bool__(self):
        r = object.__eq__(self.a, self.b)
        if r is NotImplemented:
            return False
        else:
            return True


class NotEqual(Condition, BinaryExpr):
    def __init__(self, a, b):
        super().__init__(a, b)


class LogicalAnd(Condition, BinaryExpr):
    def __init__(self, a, b):
        super().__init__(a, b)


class LogicalOr(Condition, BinaryExpr):
    def __init__(self, a, b):
        super().__init__(a, b)


class LogicalNot(Condition, UnaryExpr):
    def __init__(self, a):
        super().__init__(a)


class Neg(UnaryExpr):
    def __init__(self, a):
        super().__init__(a)


class Add(BinaryExpr):
    def __init__(self, a, b):
        super().__init__(a, b)


class Sub(BinaryExpr):
    def __init__(self, a, b):
        super().__init__(a, b)


class Multiply(BinaryExpr):
    def __init__(self, a, b):
        super().__init__(a, b)


class Div(BinaryExpr):
    def __init__(self, a, b):
        super().__init__(a, b)


class FloorDiv(BinaryExpr):
    def __init__(self, a, b):
        super().__init__(a, b)
        raise ValueError('FloorDiv is not supported in hidet by design from now on.')


class Mod(BinaryExpr):
    def __init__(self, a, b):
        super().__init__(a, b)


class BitwiseNot(UnaryExpr):
    def __init__(self, a):
        super().__init__(a)


class BitwiseAnd(BinaryExpr):
    def __init__(self, a, b):
        super().__init__(a, b)


class BitwiseOr(BinaryExpr):
    def __init__(self, a, b):
        super().__init__(a, b)


class BitwiseXor(BinaryExpr):
    def __init__(self, a, b):
        super().__init__(a, b)


class LeftShift(BinaryExpr):
    def __init__(self, a, b):
        super().__init__(a, b)


class RightShift(BinaryExpr):
    def __init__(self, a, b):
        super().__init__(a, b)


class TensorElement(Expr):
    def __init__(self, base, indices, protected=False):
        self.base: Expr = base
        self.indices: Tuple[Expr, ...] = indices
        self.protected: bool = protected

        assert isinstance(base, Expr) and isinstance(indices, tuple)
        for idx in indices:
            assert isinstance(idx, Expr)


class TensorSlice(Expr):
    def __init__(self, base, indices, starts, ends):
        # a[3, 4:, :5, :] will be represented by
        # base: a
        # indices: [3, None, None, None]
        # starts: [None, 4, None, None]
        # ends: [None, None, 5, None]
        self.base: Expr = base
        self.indices: Tuple[Optional[Expr], ...] = indices
        self.starts: Tuple[Optional[Expr], ...] = starts
        self.ends: Tuple[Optional[Expr], ...] = ends

        assert isinstance(indices, tuple) and isinstance(starts, tuple) and isinstance(ends, tuple)
        for idx in indices:
            assert idx is None or isinstance(idx, Expr)
        for start in starts:
            assert start is None or isinstance(start, Expr)
        for end in ends:
            assert end is None or isinstance(end, Expr)


class Call(Expr):
    def __init__(self, func_var, args):
        self.func_var: Var = func_var
        self.args: Tuple[Expr, ...] = args

        assert isinstance(func_var, Var) and isinstance(args, tuple)
        for arg in args:
            assert isinstance(arg, Expr)


class Let(Expr):
    def __init__(self, var, value, body):
        self.var: Expr = var
        self.value: Expr = value
        self.body: Expr = body

        assert isinstance(var, Var) and isinstance(value, Expr) and isinstance(body, Expr)


class Cast(Expr):
    def __init__(self, expr, target_type: TypeNode):
        self.expr: Expr = expr
        self.target_type: TypeNode = target_type

        assert isinstance(target_type, TypeNode)


class Constant(Expr):
    def __init__(self, value: Union[np.ndarray, float, int, complex], const_type: Union[str, DataType, TensorType]):
        from hidet.ir.dtypes import boolean

        if const_type and isinstance(const_type, str):
            const_type = data_type(const_type)
        self.value: Union[np.ndarray, float, int, complex] = value
        self.type: Union[DataType, TensorType] = const_type

        # normalize value
        if isinstance(self.type, DataType):
            if self.type.is_complex():
                self.value = complex(self.value)
            elif self.type.is_float():
                self.value = float(self.value)
            elif self.type.is_integer():
                if self.type == boolean:
                    self.value = bool(self.value)
                else:
                    self.value = int(self.value)
            elif self.type.is_vector():
                self.value = tuple(self.value)
            else:
                raise ValueError(f"Invalid data type {self.type}")
        else:
            self.value = np.array(self.value)

    def is_scalar(self) -> bool:
        return self.type and isinstance(self.type, DataType)

    def is_tensor(self) -> bool:
        return self.type and isinstance(self.type, TensorType)

    def __int__(self):
        return int(self.value)

    def __float__(self):
        return float(self.value)

    def __bool__(self):
        return bool(self.value)

    def __complex__(self):
        return complex(self.value)

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

    def __init__(self, cond: Expr, then_expr: Expr, else_expr: Expr):
        self.cond: Expr = cond
        self.then_expr: Expr = then_expr
        self.else_expr: Expr = else_expr

        assert isinstance(cond, Expr) and isinstance(then_expr, Expr) and isinstance(else_expr, Expr)


class Dereference(Expr):
    def __init__(self, expr: Expr):
        self.expr: Expr = expr

        assert isinstance(expr, Expr)


class Address(Expr):
    def __init__(self, expr: Expr):
        self.expr: Expr = expr

        assert isinstance(expr, Expr)


class Reference(Expr):
    def __init__(self, expr: Expr):
        self.expr: Expr = expr

        assert isinstance(expr, Expr)


class Var(Expr):
    id_clock = 0

    def __init__(self, hint: Optional[str], type: TypeNode, name: Optional[str] = None):
        """
        A variable may have a hint, name, and id.

        Hint is used to determine the name in codegen. Different vars may have the
        same hint. If two vars have the same hint such as 'x', the final name would be like 'x1', 'x2'.

        OUTDATED:
        Name is the determined name in the final code. Used by primitive variables such as 'threadIdx.x'. No variable
        should have a same name as primitive objects (including primitive variables and primitive functions).

        ID is used to track the allocation of Var object in python, which is only used to help us to distinguish
        different Var in python debugger.
        """
        self.hint: Optional[str] = hint
        self.name: Optional[str] = name
        self.type: Union[TypeNode, TensorType, TensorPointerType, FuncType] = type
        self.id: int = self.new_id()

    @staticmethod
    def new_id():
        Var.id_clock += 1
        return Var.id_clock

    @staticmethod
    def reset_id_counter():
        Var.id_clock = 0


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
Int = Union[int, Expr]


"""
The following are the mapping from hidet expression class to corresponding python operator.
Used by compilation-time constant folding.
"""
_operator_dict: Dict[Type[Expr], Callable] = {
    # unary arithmetic
    Neg: operator.neg,
    LogicalNot: operator.not_,
    BitwiseNot: operator.invert,
    # binary arithmetic
    Add: operator.add,
    Sub: operator.sub,
    Multiply: operator.mul,
    Div: lambda a, b: a // b if isinstance(a, int) and isinstance(b, int) else a / b,
    Mod: operator.mod,
    BitwiseOr: operator.or_,
    BitwiseAnd: operator.and_,
    BitwiseXor: operator.xor,
    Equal: operator.eq,
    NotEqual: operator.ne,
    LessThan: operator.lt,
    LessEqual: operator.le,
    LogicalAnd: lambda a, b: a and b,
    LogicalOr: lambda a, b: a or b,
    LeftShift: operator.lshift,
    RightShift: operator.rshift,
}


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
        return len(v.type.shape)
    elif isinstance(v, Constant) and isinstance(v.type, TensorType):
        return len(v.type.shape)
    elif isinstance(v, Cast) and isinstance(v.target_type, PointerType):
        return 1
    else:
        raise ValueError('Can not infer the tensor rank of "{}"'.format(v))


def tensor_slice(
    base: Expr, indices: Sequence[Optional[Int]], starts: Sequence[Optional[Int]], ends: Sequence[Optional[Int]]
) -> TensorSlice:
    """
    Create a tensor slice expression.

    Parameters
    ----------
    base: Expr
        The base tensor expression.

    indices: Sequence[Optional[Int]]
        The indices of the tensor slice.

    starts: Sequence[Optional[Int]]
        The start indices of the tensor slice.

    ends: Sequence[Optional[Int]]
        The end indices of the tensor slice.

    Returns
    -------
    ret: TensorSlice
        The tensor slice expression.
    """
    if len(indices) != len(starts) or len(indices) != len(ends):
        raise ValueError('The length of indices, starts and ends should be the same.')

    indices = tuple(convert(i) for i in indices)
    starts = tuple(convert(i) for i in starts)
    ends = tuple(convert(i) for i in ends)
    return TensorSlice(base, indices, starts, ends)


def tensor_element(base: Expr, indices: Sequence[Int], protected=False):
    indices = tuple(convert(i) for i in indices)
    return TensorElement(base, indices, protected)


def _chain_binary_op(op, operands, default):
    if len(operands) == 0:
        return convert(default)
    elif len(operands) == 1:
        return convert(operands[0])
    else:
        a = _chain_binary_op(op, operands[:-1], default)
        b = convert(operands[-1])
        return op(a, b)


def logical_and(*args: Union[Expr, bool]) -> LogicalAnd:
    return _chain_binary_op(LogicalAnd, args, True)


def logical_or(*args: Union[Expr, bool]) -> LogicalOr:
    return _chain_binary_op(LogicalOr, args, False)


def logical_not(a: Union[Expr, PyScalar]) -> LogicalNot:
    a = convert(a)
    return LogicalNot(a)


def equal(a: Union[Expr, PyScalar], b: Union[Expr, PyScalar]) -> Equal:
    a = convert(a)
    b = convert(b)
    return Equal(a, b)


def less_than(a: Union[Expr, PyScalar], b: Union[Expr, PyScalar]) -> LessThan:
    a = convert(a)
    b = convert(b)
    return LessThan(a, b)


def less_equal(a: Union[Expr, PyScalar], b: Union[Expr, PyScalar]) -> LessEqual:
    a = convert(a)
    b = convert(b)
    return LessEqual(a, b)


def not_equal(a: Union[Expr, PyScalar], b: Union[Expr, PyScalar]) -> NotEqual:
    a = convert(a)
    b = convert(b)
    return NotEqual(a, b)


def left_shift(a: Union[Expr, int], b: Union[Expr, int]) -> LeftShift:
    a = convert(a)
    b = convert(b)
    return LeftShift(a, b)


def right_shift(a: Union[Expr, int], b: Union[Expr, int]) -> RightShift:
    a = convert(a)
    b = convert(b)
    return RightShift(a, b)


def bitwise_and(*args: Union[Expr, int]) -> BitwiseAnd:
    return _chain_binary_op(BitwiseAnd, args, -1)


def bitwise_or(*args: Union[Expr, int]) -> BitwiseOr:
    return _chain_binary_op(BitwiseOr, args, 0)


def bitwise_xor(*args: Union[Expr, int]) -> BitwiseXor:
    return _chain_binary_op(BitwiseXor, args, 0)


def cast(v: Expr, dtype: Union[str, DataType, TypeNode]):
    if not isinstance(v, Expr):
        raise ValueError('Expect an expression, got {}'.format(type(v).__name__))

    if isinstance(dtype, str):
        dtype = data_type(dtype)

    if isinstance(v, Constant):
        return Constant(v.value, dtype)
    else:
        return Cast(v, dtype)


def call(func: Var, args: Sequence[Union[Expr, PyScalar]]) -> Call:
    args = tuple(convert(a) for a in args)
    return Call(func, args)


def const_tensor(value: np.ndarray) -> Constant:
    from hidet.ir.utils.type_utils import from_numpy_dtype

    return Constant(value=value, const_type=tensor_type(dtype=from_numpy_dtype(value.dtype), shape=list(value.shape)))


def tensor_pointer_var(hint: str, shape=None, dtype: Union[str, DataType] = 'float32', layout=None):
    return Var(hint, tensor_pointer_type(dtype=dtype, shape=shape, layout=layout))


def view(ptr: Expr, tp: TensorType) -> Expr:
    if not isinstance(tp, TensorType):
        raise ValueError('Expect a tensor type, got {}'.format(type(tp).__name__))
    return cast(ptr, TensorPointerType.from_tensor_type(tp))


def address(v: Expr) -> Expr:
    return Address(v)


def deref(v: Expr) -> Expr:
    return Dereference(v)
