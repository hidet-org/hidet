import string
from typing import Optional, Union, Sequence
from .node import Node
from .type import TypeNode, TensorType, TensorType, ScalarType, Scope, tensor_type, scalar_type

PyScalar = Union[int, float]


class Expr(Node):
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

    def __eq__(self, other):
        return Equal(self, other)

    def __ge__(self, other):
        return LessEqual(other, self)

    def __invert__(self):
        from hidet.ir.dialects.lowlevel import Address
        return Address(self)

    def __getitem__(self, item):
        if not isinstance(item, (tuple, list)):
            item = [item]
        indices = [idx if not isinstance(idx, slice) else None for idx in item]
        slices = [idx for idx in item if isinstance(idx, slice)]
        assert len(slices) == 0
        return TensorElement(self, indices)

    def __hash__(self):
        return id(self)

    def __int__(self):
        return int(self)

    def __float__(self):
        return float(self)

    def __str__(self):
        from hidet.ir.functors import astext
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


class TensorElement(Expr):
    def __init__(self, base, indices):
        self.base = base
        self.indices = convert(indices)


class Call(Expr):
    def __init__(self, func, args):
        # todo: use function name (str) directly as the function identity
        self.func_var: Var = func
        self.args = convert(args)


class Let(Expr):
    def __init__(self, var, value, body):
        self.var = var
        self.value = convert(value)
        self.body = convert(body)


class Constant(Expr):
    def __init__(self, value, dtype=None):
        self.value = value
        if dtype and isinstance(dtype, str):
            dtype = ScalarType(dtype)
        self.dtype: Optional[ScalarType] = dtype

    def __int__(self):
        return int(self.value)

    def __float__(self):
        return float(self.value)


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
        if hint is not None:
            assert not is_reserved_name(hint)
        self.hint = hint
        self.name = name
        self.type: TypeNode = type
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


def tensor_var(hint: str, shape, scope: str = 'global', dtype: str = 'float32', layout=None) -> Var:
    return Var(hint, tensor_type(scope, dtype, shape, layout))


def is_one(v: Expr) -> bool:
    return isinstance(v, Constant) and v.value == 1


def is_zero(v: Expr) -> bool:
    return isinstance(v, Constant) and v.value == 0


def is_true(v: Expr) -> bool:
    return isinstance(v, Constant) and v.dtype.name == 'bool' and v.value is True


def is_false(v: Expr) -> bool:
    return isinstance(v, Constant) and v.dtype.name == 'bool' and v.value is False


def is_const_int(v: Expr) -> bool:
    return isinstance(v, Constant) and v.dtype.name == 'int32'


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
