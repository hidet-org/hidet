from hidet.ir.type import ScalarType
from hidet.ir.expr import Axis, Expr, Constant, convert


class ComputeNode(Expr):
    def __init__(self, name):
        self.name = name


class ScalarInput(ComputeNode):
    def __init__(self, name, dtype):
        super().__init__(name)
        self.dtype = dtype


class TensorInput(ComputeNode):
    def __init__(self, name, dtype, shape):
        super().__init__(name)
        self.dtype = dtype
        self.shape = shape


class TensorCompute(ComputeNode):
    def __init__(self, name, shape, axes, value):
        super().__init__(name)
        self.shape = [convert(v) for v in shape]
        self.axes = axes
        self.value = value


class ReduceCompute(ComputeNode):
    def __init__(self, value, axis, reduce_type):
        super().__init__(None)
        self.value = value
        self.axis = axis
        self.reduce_type = reduce_type

    def init_const(self):
        if self.reduce_type == 'sum':
            return Constant(0.0, ScalarType('float32'))
        else:
            raise NotImplementedError()

    def combine(self, lhs, rhs):
        if self.reduce_type == 'sum':
            return lhs + rhs
        else:
            raise NotImplementedError()


def scalar_input(name, dtype):
    if isinstance(dtype, str):
        dtype = ScalarType(dtype)
    assert isinstance(dtype, ScalarType)
    return ScalarInput(name, dtype)


def tensor_input(name, base_type, shape):
    if isinstance(base_type, str):
        base_type = ScalarType(base_type)
    assert isinstance(base_type, ScalarType)
    shape = [convert(s) for s in shape]
    return TensorInput(name, base_type, shape)


def reduce_sum(expr, axis):
    assert isinstance(axis, Axis)
    expr = convert(expr)
    return ReduceCompute(expr, axis, 'sum')


def compute(name, shape, fcompute):
    shape = [convert(v) for v in shape]
    axes = [Axis(s) for s in shape]
    value = convert(fcompute(*axes))
    return TensorCompute(name, shape, axes, value)

