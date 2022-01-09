from typing import Optional, Union
from .node import Node
from .type import BaseType, TensorType, TensorType, ScalarType, Scope, tensor_type, scalar_type


class Expr(Node):
    def __add__(self, other):
        return Add(self, other)

    def __sub__(self, other):
        return Sub(self, other)

    def __mul__(self, other):
        return Multiply(self, other)

    def __truediv__(self, other):
        return Div(self, other)

    def __floordiv__(self, other):
        return FloorDiv(self, other)

    def __mod__(self, other):
        return Mod(self, other)

    def __lt__(self, other):
        return LessThan(self, other)

    def __eq__(self, other):
        return Equal(self, other)

    def __getitem__(self, item):
        if not isinstance(item, tuple):
            item = [item]
        if any(isinstance(v, slice) for v in item):
            return TensorSlice(self, item)
        else:
            return TensorElement(self, item)

    def __hash__(self):
        return object.__hash__(self)

    def __str__(self):
        from hidet.ir.functors import astext
        return str(astext(self))


class BinaryOp(Expr):
    def __init__(self, a, b):
        self.a = convert(a)
        self.b = convert(b)


def convert(obj):
    if isinstance(obj, Expr):
        return obj
    elif isinstance(obj, int):
        return Constant(obj, ScalarType('int32'))
    elif isinstance(obj, float):
        return Constant(obj, ScalarType('float32'))
    else:
        raise NotImplementedError()


class Condition(Expr):
    pass


class LessThan(Condition, BinaryOp):
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


class TensorSlice(Expr):
    def __init__(self, base, indices, starts, ends):
        self.base = base
        self.indices = indices
        self.starts = starts
        self.ends = ends


class TensorElement(Expr):
    def __init__(self, base, indices):
        self.base = base
        self.indices = [convert(idx) for idx in indices]


class Call(Expr):
    def __init__(self, func, args):
        self.func_var: Var = func
        self.args = args


class Constant(Expr):
    def __init__(self, value, dtype=None):
        self.value = value
        self.dtype: ScalarType = dtype


class Var(Expr):
    id_clock = 0

    def __init__(self, hint: Optional[str], type: BaseType):
        self.hint = hint
        self.type = type
        self.id = self.new_id()

    @staticmethod
    def new_id():
        Var.id_clock += 1
        return Var.id_clock

    @staticmethod
    def reset_id_counter():
        Var.id_clock = 0


class Axis(Var):
    def __init__(self, extent, min_value=0):
        super().__init__(None, ScalarType('int32'))
        self.min_value = convert(min_value)
        self.extent = convert(extent)


class IntVar(Var):
    def __init__(self, name=None, extent=None, min_value=0):
        super().__init__(name, ScalarType('int32'))
        self.name = name
        self.min_value = min_value
        self.extent = extent


def var(hint: str, scope: str = 'global', dtype: str = 'float32', shape=None, strides=None):
    if shape is None or len(shape) == 0:
        type = ScalarType(dtype)
    else:
        type = TensorType(Scope(scope), ScalarType(dtype), shape, strides)
    return Var(hint, type)


def scalar_var(hint: str, dtype: Union[str, ScalarType] = 'float32'):
    dtype = dtype if isinstance(dtype, ScalarType) else scalar_type(dtype)
    return Var(hint, dtype)


def tensor_var(hint: str, shape, scope: str = 'global', dtype: str = 'float32', strides=None):
    return Var(hint, tensor_type(scope, dtype, shape, strides))


def is_one(v: Expr):
    return isinstance(v, Constant) and v.value == 1


def is_zero(v: Expr):
    return isinstance(v, Constant) and v.value == 0

