from typing import Union, Sequence, Tuple, Optional, List, Dict, Any
from hidet.ir.type import ScalarType, TensorType, Scope, tensor_type, scalar_type
from hidet.ir.expr import Expr, Constant, convert, Var, var, And, if_then_else
from hidet.utils.info import float_type_min_value


class ComputeNode(Expr):
    def __init__(self, name):
        self.name: Optional[str] = name


class ScalarNode(ComputeNode):
    def __init__(self, name, data_type, reduce_compute=None):
        super().__init__(name)
        self.data_type: ScalarType = data_type
        self.reduce_compute: Optional[ReduceCompute] = reduce_compute

    def is_input(self) -> bool:
        return self.reduce_compute is None


class TensorNode(ComputeNode):
    def __init__(self, name, data_type, grid_compute=None):
        super().__init__(name)
        self.data_type: TensorType = data_type
        self.grid_compute: Optional[GridCompute] = grid_compute

    def is_input(self) -> bool:
        return self.grid_compute is None

    def const_shape(self) -> List[int]:
        return self.data_type.const_shape()

    def protect_read(self, indices, default_value=0.0) -> Expr:
        conds = []
        assert len(indices) == len(self.data_type.shape)
        for index, extent in zip(indices, self.data_type.shape):
            conds.append(0 <= index)
            conds.append(index < extent)
        return if_then_else(And.join(*conds), self.__getitem__(indices), default_value)


class GridCompute:
    def __init__(self, shape, axes, value):
        from hidet.ir.functors import collect, simplify
        self.input_tensors: List[TensorNode] = collect(value, TensorNode, stop_when_found=True)
        self.input_scalars: List[ScalarNode] = collect(value, ScalarNode, stop_when_found=True)
        self.shape: Tuple[Expr] = convert(shape)
        self.axes: Tuple[Var] = convert(axes)
        self.value: Expr = simplify(value)


class ReduceCompute:
    def __init__(self, shape, axes, value, reduce_type, accumulate_dtype: str = 'float32'):
        from hidet.ir.functors import collect, simplify
        self.input_tensors: List[TensorNode] = collect(value, TensorNode, stop_when_found=True)
        self.input_scalars: List[ScalarNode] = collect(value, ScalarNode, stop_when_found=True)
        self.shape: Tuple[Expr] = convert(shape)
        self.axes: Tuple[Var] = convert(axes)
        self.value: Expr = simplify(value)
        self.reduce_type: str = reduce_type
        self.accumulate_dtype = accumulate_dtype
        assert reduce_type in ['max', 'avg', 'sum']

    @staticmethod
    def init_const(reduce_type: str, data_type: Union[ScalarType, str]):
        init_dict = {
            'sum': Constant(0.0, data_type),
            'avg': Constant(0.0, data_type),
            'max': Constant(float_type_min_value(), data_type)
        }
        return init_dict[reduce_type]

    @staticmethod
    def combine(reduce_type: str, lhs: Expr, rhs: Expr):
        import hidet.ir.primitives as primitives
        func_dict = {
            'sum': lambda a, b: a + b,
            'avg': lambda a, b: a + b,
            'max': lambda a, b: primitives.max(a, b)
        }
        return func_dict[reduce_type](lhs, rhs)

    @staticmethod
    def finalize(reduce_type: str, acc: Expr, size: Expr):
        func_dict = {
            'sum': lambda acc, size: acc,
            'avg': lambda acc, size: acc / size,
            'max': lambda acc, size: acc
        }
        return func_dict[reduce_type](acc, size)

    def const_shape(self) -> List[int]:
        return [int(v) for v in self.shape]


def scalar_input(name, dtype):
    if isinstance(dtype, str):
        dtype = ScalarType(dtype)
    else:
        assert isinstance(dtype, ScalarType)
    return ScalarNode(name, dtype, reduce_compute=None)


def tensor_input(name, base_type, shape, scope=None, layout=None):
    data_type = tensor_type(scope, base_type, shape, layout)
    return TensorNode(name, data_type, grid_compute=None)


def reduce(shape: Sequence[Union[int, Expr]], fcompute, reduce_type: str, accumulate_dtype: str = 'float32') -> ScalarNode:
    from hidet.ir.functors import infer_type
    shape = convert(shape)
    axes = [var() for _ in shape]
    value = convert(fcompute(*axes))
    return ScalarNode(
        name=f'acc_{reduce_type}',
        data_type=infer_type(value),
        reduce_compute=ReduceCompute(shape, axes, value, reduce_type, accumulate_dtype)
    )


def compute(name, shape, fcompute, scope=None, layout=None) -> TensorNode:
    from hidet.ir.functors import infer_type
    shape = convert(shape)
    axes = [var() for _ in shape]
    value = convert(fcompute(*axes))
    if scope is None:
        # todo: automatic determine scope by checking inputs' scope
        scope = 'global'
    return TensorNode(
        name=name,
        data_type=tensor_type(scope, dtype=infer_type(value), shape=shape, layout=layout),
        grid_compute=GridCompute(shape, axes, value)
    )

