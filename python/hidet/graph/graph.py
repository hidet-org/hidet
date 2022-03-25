from typing import List, Optional, Dict
from hidet.ir.type import TensorType, ScalarType
from hidet.ir.layout import DataLayout
from hidet.ir.task import Task
from hidet.runtime.value import TensorValue, randn


class Operator:
    def __init__(self, inputs, task):
        self.inputs: List[Tensor] = inputs
        self.task: Task = task
        self.outputs: List[Tensor] = []
        output = Tensor.from_type(
            ttype=task.type_of_param(task.compute),
            op=self,
            index=0
        )
        self.outputs.append(output)


def convert(v):
    if isinstance(v, float):
        return Tensor(dtype='float32', shape=[1], name='scalar_const')
    elif isinstance(v, Tensor):
        return v
    else:
        raise NotImplementedError()


class Tensor:
    def __init__(self, dtype, shape, layout=None, op=None, index=0, name=None, value=None, init_method=None):
        self.op: Optional[Operator] = op
        self.index: int = index
        self.dtype: ScalarType = ScalarType(dtype) if isinstance(dtype, str) else dtype
        self.shape: List[int] = shape
        self.layout: DataLayout = layout if layout else DataLayout.row_major(shape)
        self.name: str = name
        self.value: Optional[TensorValue] = value
        if value is None and init_method is not None:
            assert init_method == 'rand'
            self.value = randn(shape=shape, scalar_type='float32', scope='global')

    @staticmethod
    def from_type(ttype: TensorType, op, index) -> 'Tensor':
        return Tensor(op, index, ttype.scalar_type, ttype.shape, ttype.layout)

    def __add__(self, other):
        from .ops import add
        return add(self, convert(other))

    def __sub__(self, other):
        from .ops import sub
        return sub(self, convert(other))

    def __mul__(self, other):
        from .ops import multiply
        return multiply(self, convert(other))

    def __truediv__(self, other):
        from .ops import divide
        return divide(self, convert(other))


class Module:
    def __init__(self):
        self.name = None
        self.parameters: Dict[str, Tensor] = {}
        self.submodules: Dict[str, Module] = {}

    def __setattr__(self, key, value):
        if isinstance(value, Tensor):
            value.name = key
            self.parameters[key] = value
        elif isinstance(value, Module):
            value.name = '{}.{}'.format(self.name, key) if self.name else key
            self.submodules[key] = value
        elif value is None and (key in self.parameters or key in self.submodules):
            if key in self.parameters:
                del self.parameters[key]
            if key in self.submodules:
                del self.submodules
        else:
            super().__setattr__(self, key, value)
        cnt = sum([1 for collection in [self.parameters, self.submodules, self.__dict__] if key in collection])
        assert cnt <= 1, 'duplicated definition of {}'.format(key)

    def __getattr__(self, item):
        if item in self.parameters:
            return self.parameters[item]
        if item in self.submodules:
            return self.submodules[item]
        raise AttributeError(item)

    def forward(self, *args):
        raise NotImplementedError()

