from typing import Union, Sequence, Tuple, Optional, List
from hidet.ir.type import ScalarType, TensorType
from hidet.ir.expr import Expr, Constant, convert, Var, var, And, if_then_else
from hidet.utils.info import float_type_min_value
from hidet.ir.layout import DataLayout


class ComputeNode(Expr):
    def __init__(self, name):
        self.name = name

    def data_type(self) -> Union[ScalarType, TensorType]:
        from hidet.ir.functors import infer_type
        return infer_type(self)


class ScalarInput(ComputeNode):
    def __init__(self, name, dtype):
        super().__init__(name)
        if dtype is not None and isinstance(dtype, str):
            dtype = ScalarType(dtype)
        self.dtype = dtype

    def data_type(self) -> Union[ScalarType, TensorType]:
        return self.dtype


class TensorInput(ComputeNode):
    def __init__(self, name, dtype=None, shape=None, scope: str = None, layout: DataLayout = None):
        super().__init__(name)
        if dtype and isinstance(dtype, str):
            dtype = ScalarType(dtype)
        self.dtype: ScalarType = dtype
        self.shape: Tuple[Union[int, Expr]] = convert(shape)
        self.scope: str = scope
        self.layout: DataLayout = layout

    def const_shape(self) -> List[int]:
        return [int(v) for v in self.shape]

    def protect_read(self, indices, default_value=0.0):
        conds = []
        assert len(indices) == len(self.shape)
        for index, extent in zip(indices, self.shape):
            conds.append(0 <= index)
            conds.append(index < extent)
        return if_then_else(And.join(*conds), self.__getitem__(indices), default_value)


class TensorCompute(ComputeNode):
    def __init__(self, name, shape, axes, value, accumulate: str = None, predicate: Optional[Expr] = None, scope: str = None, layout: DataLayout = None):
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
        self.scope = scope
        self.layout = layout


class ReduceCompute(ComputeNode):
    def __init__(self, value, shape, axes, reduce_type):
        super().__init__(None)
        self.value: Expr = value
        self.axes: Tuple[Var] = convert(axes)
        self.shape: Tuple[Expr] = convert(shape)
        self.reduce_type: str = reduce_type
        assert len(self.axes) == len(self.shape)

    def init_const(self):
        init_dict = {
            'sum': Constant(0.0, ScalarType('float32')),
            'avg': Constant(0.0, ScalarType('float32')),
            'max': Constant(float_type_min_value(), ScalarType('float32'))
        }
        return init_dict[self.reduce_type]

    def combine(self, lhs, rhs):
        from hidet.ir.primitives.func import cuda_max
        func_dict = {
            'sum': lambda a, b: a + b,
            'avg': lambda a, b: a + b,
            'max': lambda a, b: cuda_max(a, b)
        }
        return func_dict[self.reduce_type](lhs, rhs)

    def finalize(self, acc, size):
        func_dict = {
            'sum': lambda acc, size: acc,
            'avg': lambda acc, size: acc / size,
            'max': lambda acc, size: acc
        }
        return func_dict[self.reduce_type](acc, size)


class CustomCompute(ComputeNode):
    def __init__(self, name, data_type: Union[ScalarType, TensorType]):
        super().__init__(name)
        self.name = name
        self.out_data_type = data_type


def scalar_input(name, dtype):
    if isinstance(dtype, str):
        dtype = ScalarType(dtype)
    assert isinstance(dtype, ScalarType)
    return ScalarInput(name, dtype)


def tensor_input(name, base_type, shape, scope=None, layout=None):
    return TensorInput(name, base_type, shape, scope, layout)


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


def compute(name, shape, fcompute, accumulate=None, predicate=None, scope=None, layout=None):
    shape = [convert(v) for v in shape]
    axes = [var() for _ in shape]
    value = convert(fcompute(*axes))
    if predicate is not None:
        predicate = convert(predicate(*axes))
    return TensorCompute(name, shape, axes, value, accumulate, predicate, scope, layout)


def custom_compute(name, data_type):
    return CustomCompute(name, data_type)
