from typing import Union, Sequence, Tuple, Optional, List, Dict, Any
from hidet.ir.type import ScalarType, TensorType, Scope, tensor_type, scalar_type
from hidet.ir.expr import Expr, Constant, convert, Var, var, And, if_then_else
from hidet.utils.info import float_type_min_value
from hidet.ir.layout import DataLayout


class ComputeNode(Expr):
    def __init__(self, name, data_type):
        self.name: Optional[str] = name
        self.data_type: Optional[Union[ScalarType, TensorType]] = data_type


class ScalarInput(ComputeNode):
    def __init__(self, name, data_type):
        super().__init__(name, data_type)


class TensorInput(ComputeNode):
    def __init__(self, name, data_type):
        super().__init__(name, data_type)

    def const_shape(self) -> List[int]:
        return [int(v) for v in self.data_type.shape]

    def protect_read(self, indices, default_value=0.0):
        conds = []
        assert len(indices) == len(self.data_type.shape)
        for index, extent in zip(indices, self.data_type.shape):
            conds.append(0 <= index)
            conds.append(index < extent)
        return if_then_else(And.join(*conds), self.__getitem__(indices), default_value)


class TensorCompute(ComputeNode):
    def __init__(self, name, shape, axes, value, data_type, accumulate: str = None):
        """
        accumulate: Optional[str], choices: 'sum', None
            The accumulate method. Let value be the value of corresponding element in output grid.
            - None: dest = value
            - 'sum': dest = dest + value
        """
        super().__init__(name, data_type)
        self.shape: Tuple[Expr] = convert(shape)
        self.axes: Tuple[Var] = convert(axes)
        self.value: Expr = value
        self.accumulate: Optional[str] = accumulate

    def const_shape(self) -> List[int]:
        return [int(v) for v in self.data_type.shape]


class ReduceCompute(ComputeNode):
    def __init__(self, value, shape, axes, reduce_type, data_type):
        super().__init__(name=None, data_type=data_type)
        self.value: Expr = value
        self.axes: Tuple[Var] = convert(axes)
        self.shape: Tuple[Expr] = convert(shape)
        self.reduce_type: str = reduce_type
        assert len(self.axes) == len(self.shape)

    def const_shape(self) -> List[int]:
        return [int(v) for v in self.shape]

    def init_const(self):
        init_dict = {
            'sum': Constant(0.0, self.data_type),
            'avg': Constant(0.0, self.data_type),
            'max': Constant(float_type_min_value(), self.data_type)
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
    def __init__(self, name, identifier: str, params: Optional[List[Union[ComputeNode]]], data_type: Optional[Union[ScalarType, TensorType]], attributes=None):
        super().__init__(name, data_type)
        self.name: str = name
        self.identifier: str = identifier
        self.params: Optional[List[Union[ComputeNode]]] = params
        self.attributes: Optional[Dict[str, Any]] = attributes
        if self.attributes is not None:
            assert isinstance(self.attributes, dict)
            self.attributes = {key: convert(value) for key, value in self.attributes.items()}


def scalar_input(name, dtype):
    if isinstance(dtype, str):
        dtype = ScalarType(dtype)
    assert isinstance(dtype, ScalarType)
    return ScalarInput(name, dtype)


def tensor_input(name, base_type, shape, scope=None, layout=None):
    data_type = tensor_type(scope, base_type, shape, layout)
    return TensorInput(name, data_type)


def reduce(shape: Sequence[Union[int, Expr]], fcompute, reduce_type: str):
    shape = convert(shape)
    axes = [var() for _ in shape]
    value = convert(fcompute(*axes))
    return ReduceCompute(value, shape, axes, reduce_type, scalar_type('float32'))


def compute(name, shape, fcompute, accumulate=None, scope=None, layout=None):
    from hidet.ir.functors import infer_type
    shape = [convert(v) for v in shape]
    axes = [var() for _ in shape]
    value = convert(fcompute(*axes))
    if scope is None:
        scope = Scope('temp')
    data_type = tensor_type(scope, dtype=infer_type(value), shape=shape, layout=layout)
    return TensorCompute(name, shape, axes, value, data_type, accumulate)


def custom_compute(name, identifier, params, data_type, attributes=None):
    return CustomCompute(name, identifier, params, data_type, attributes)
