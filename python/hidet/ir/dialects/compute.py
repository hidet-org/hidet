from typing import Union, Sequence, Tuple, Optional
from hidet.ir.type import ScalarType
from hidet.ir.expr import Expr, Constant, convert, Var, var, And, if_then_else


class ComputeNode(Expr):
    def __init__(self, name):
        self.name = name


class ScalarInput(ComputeNode):
    def __init__(self, name, dtype):
        super().__init__(name)
        if dtype is not None and isinstance(dtype, str):
            dtype = ScalarType(dtype)
        self.dtype = dtype


class TensorInput(ComputeNode):
    def __init__(self, name, dtype=None, shape=None):
        super().__init__(name)
        if dtype and isinstance(dtype, str):
            dtype = ScalarType(dtype)
        self.dtype: ScalarType = dtype
        self.shape = shape

    def protect_read(self, indices, default_value=0.0):
        conds = []
        assert len(indices) == len(self.shape)
        for index, extent in zip(indices, conds):
            conds.append(0 <= index)
            conds.append(index < extent)
        return if_then_else(And.join(conds), self.__getitem__(indices), default_value)


class TensorCompute(ComputeNode):
    def __init__(self, name, shape, axes, value, accumulate: str = None, predicate: Optional[Expr] = None):
        """
        accumulate: Optional[str], choices: 'sum', None
            The accumulate method. Let value be the value of corresponding element in output grid.
            - None: dest = value
            - 'sum': dest = dest + value
        """
        super().__init__(name)
        self.shape: Tuple[Expr] = convert(shape)
        self.axes: Tuple[Var] = convert(axes)
        self.value: Expr = value
        self.accumulate: Optional[str] = accumulate
        self.predicate: Optional[Expr] = predicate


class ReduceCompute(ComputeNode):
    def __init__(self, value, shape, axes, reduce_type):
        super().__init__(None)
        self.value: Expr = value
        self.axes: Tuple[Var] = convert(axes)
        self.shape: Tuple[Expr] = convert(shape)
        self.reduce_type: str = reduce_type
        assert len(self.axes) == len(self.shape)

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


def reduce_sum(expr, axes, shape: Union[Sequence[Union[int, Expr]], Union[int, Expr]]):
    if not isinstance(axes, (tuple, list)):
        axes = [axes]
    assert all(isinstance(axis, Var) for axis in axes)
    if not isinstance(shape, (tuple, list)):
        shape = [shape]
    shape = [convert(v) for v in shape]
    expr = convert(expr)
    return ReduceCompute(expr, shape, axes, 'sum')


def reduce(shape: Sequence[Union[int, Expr]], fcompute, reduce_type: str):
    shape = convert(shape)
    axes = [var() for _ in shape]
    value = convert(fcompute(*axes))
    return ReduceCompute(value, shape, axes, reduce_type)


def compute(name, shape, fcompute, accumulate=None, predicate=None):
    shape = [convert(v) for v in shape]
    axes = [var() for _ in shape]
    value = convert(fcompute(*axes))
    if predicate is not None:
        predicate = convert(predicate(*axes))
    return TensorCompute(name, shape, axes, value, accumulate, predicate)
