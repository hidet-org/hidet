import string
import numpy as np
from typing import Optional, Union, Sequence, List, Tuple
from .node import Node
from .type import TypeNode, TensorType, TensorType, ScalarType, Scope, tensor_type, scalar_type
from .layout import DataLayout

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
        return FloorDiv(self, other)

    def __rfloordiv__(self, other):
        return FloorDiv(other, self)

    def __mod__(self, other):
        return Mod(self, other)

    def __rmod__(self, other):
        return Mod(other, self)

    def __lt__(self, other):
        return LessThan(self, other)

    def __le__(self, other):
        return LessEqual(self, other)

    #
    # for performance, we should use Equal(e1, e2) to represent equivalence expression
    #
    # def __eq__(self, other):
    #     return Equal(self, other)
    #
    # def __hash__(self):
    #     return id(self)
    #

    def __ge__(self, other):
        return LessEqual(other, self)

    def __invert__(self):
        from hidet.ir.dialects.lowlevel import Address
        return Address(self)

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

    def __int__(self):
        assert isinstance(self, Constant), 'Expect a Constant, got {} with type {}'.format(self, type(self))
        return int(self)

    def __float__(self):
        assert isinstance(self, Constant), 'Expect a Constant, got {} with type {}'.format(self, type(self))
        return float(self)

    def __str__(self):
        from hidet.ir.functors import astext
        # return str(astext(self)) + ' at {}'.format(hex(id(self)))
        return str(astext(self))

    def is_const(self):
        return isinstance(self, Constant)

    def const(self) -> 'Constant':
        assert isinstance(self, Constant)
        return self


class BinaryOp(Expr):
    def __init__(self, a, b):
        self.a = convert(a)
        self.b = convert(b)


class UnaryOp(Expr):
    def __init__(self, a):
        self.a = convert(a)


def convert(obj: Optional[Union[Expr, PyScalar, tuple, Sequence]]) -> Optional[Union[Expr, tuple]]:
    if isinstance(obj, Expr):
        return obj
    elif isinstance(obj, bool):
        return Constant(obj, ScalarType('bool'))
    elif isinstance(obj, int):
        return Constant(obj, ScalarType('int32'))
    elif isinstance(obj, float):
        return Constant(obj, ScalarType('float32'))
    elif isinstance(obj, (tuple, list)):
        return tuple([convert(v) for v in obj])
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


class And(Condition, BinaryOp):
    def __init__(self, a, b):
        super().__init__(a, b)

    @staticmethod
    def join(*conds):
        cond = convert(True)
        for c in conds:
            cond = And(cond, convert(c))
        return cond

    @staticmethod
    def join_list(conds: Sequence[Condition]):
        return And.join(*conds)


class Or(Condition, BinaryOp):
    def __init__(self, a, b):
        super().__init__(a, b)

    @staticmethod
    def join(*conds):
        cond = convert(False)
        for c in conds:
            cond = Or(cond, convert(c))
        return cond


class Not(Condition, UnaryOp):
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
    def __init__(self, base, indices):
        self.base = base
        self.indices = convert(indices)


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
    def __init__(self, func, args):
        self.func_var: Var = func
        self.args = convert(args)


class Let(Expr):
    def __init__(self, var, value, body):
        self.var = var
        self.value = convert(value)
        self.body = convert(body)


class Cast(Expr):
    def __init__(self, expr, target_type):
        self.expr = expr
        if isinstance(target_type, str):
            target_type = ScalarType(target_type)
        self.target_type = target_type


class Constant(Expr):
    def __init__(self, value=None, data_type=None):
        if data_type and isinstance(data_type, str):
            data_type = ScalarType(data_type)
        self.value: Optional[np.ndarray, float, int] = value
        self.data_type: Optional[Union[ScalarType, TensorType]] = data_type

    def is_scalar(self) -> bool:
        return self.data_type and isinstance(self.data_type, ScalarType)

    def is_tensor(self) -> bool:
        return self.data_type and isinstance(self.data_type, TensorType)

    def __int__(self):
        return int(self.value)

    def __float__(self):
        return float(self.value)

    def array(self) -> np.ndarray:
        return self.value


class IfThenElse(Expr):
    def __init__(self, cond: Union[Expr, PyScalar], then_expr: Union[Expr, PyScalar], else_expr: Union[Expr, PyScalar]):
        self.cond = convert(cond)
        self.then_expr = convert(then_expr)
        self.else_expr = convert(else_expr)


class Var(Expr):
    id_clock = 0

    def __init__(self, hint: Optional[str], type: TypeNode, name: Optional[str] = None):
        """
        A variable may have a hint, name, and id.

        Hint is used to determine the name in codegen. Different vars may have the
        same hint. If two vars have the same hint such as 'x', the final name would be like 'x1', 'x2'.

        Name is the determined name in the final code. Used by primitive varaibles such as 'threadIdx.x'. No variable should have
        a same name as primitive objects (including primitive variables and primitive functions).

        Id is used to track the allocation of Var object in python, which is only used to help us to distinguish different Var
        in python debugger.
        """
        from hidet.ir.primitives import is_reserved_name
        from hidet.ir.dialects.lowlevel import TensorPointerType
        if hint is not None:
            assert not is_reserved_name(hint)
        self.hint = hint
        self.name = name
        self.type: Union[TypeNode, TensorType, TensorPointerType] = type
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
    return Var(hint, ScalarType(dtype))


def scalar_var(hint: str, dtype: Union[str, ScalarType] = 'float32') -> Var:
    dtype = dtype if isinstance(dtype, ScalarType) else scalar_type(dtype)
    return Var(hint, dtype)


def tensor_var(hint: str, shape, scope: str = 'global', dtype: Union[str, ScalarType] = 'float32', layout=None) -> Var:
    return Var(hint, tensor_type(scope, dtype, shape, layout))


def is_one(v: Expr) -> bool:
    return isinstance(v, Constant) and v.value == 1


def is_zero(v: Expr) -> bool:
    return isinstance(v, Constant) and v.value == 0


def is_true(v: Expr) -> bool:
    return isinstance(v, Constant) and v.data_type.name == 'bool' and v.value is True


def is_false(v: Expr) -> bool:
    return isinstance(v, Constant) and v.data_type.name == 'bool' and v.value is False


def is_const_int(v: Expr) -> bool:
    return isinstance(v, Constant) and v.data_type.name == 'int32'


def if_then_else(cond: Union[Expr, PyScalar], then_expr: Union[Expr, PyScalar], else_expr: Union[Expr, PyScalar]) -> IfThenElse:
    return IfThenElse(convert(cond), convert(then_expr), convert(else_expr))


def is_tensor(v: Expr) -> bool:
    from hidet.ir.dialects.lowlevel import TensorPointerType
    if not isinstance(v, Var):
        return False
    return isinstance(v.type, (TensorType, TensorPointerType))


def get_tensor_layout(v: Expr):
    from hidet.ir.dialects.lowlevel import TensorPointerType
    assert isinstance(v, Var) and isinstance(v.type, (TensorType, TensorPointerType))
    return v.type.layout if isinstance(v.type, TensorType) else v.type.tensor_type.layout


def tensor_rank(v: Expr) -> int:
    from hidet.ir.dialects.compute import TensorNode
    from hidet.ir.dialects.lowlevel import TensorPointerType, PointerType
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
        return sum([1 if i is None else 0 for i in v.indices])
    elif isinstance(v, TensorNode):
        return len(v.data_type.shape)
    elif isinstance(v, Constant) and isinstance(v.data_type, TensorType):
        return len(v.data_type.shape)
    else:
        raise ValueError(v)


def cast(v: Expr, dtype):
    return Cast(v, dtype)


def const_tensor(value: np.ndarray, data_type=None) -> Constant:
    if data_type is None:
        data_type = tensor_type(
            scope='host',
            dtype=ScalarType.from_numpy_dtype(value.dtype),
            shape=list(value.shape)
        )
    return Constant(value=value, data_type=data_type)
