from typing import List, Optional, Dict
from collections import OrderedDict
from hidet.ir.type import TensorType, ScalarType
from hidet.ir.layout import DataLayout
from hidet.ir.task import Task
from hidet.runtime.value import TensorValue, randn
from hidet.tos.utils import imperative_run


class Operator:
    imperative_mode = 1
    imperative_opt_mode = 2
    lazy_mode = 3

    current_mode = imperative_mode

    def __init__(self, inputs: List['Tensor'], task: Task):
        self.inputs: List[Tensor] = inputs
        self.task: Task = task
        self.outputs: List[Tensor] = []
        if self.current_mode == self.imperative_mode:
            self.outputs.extend(imperative_run(task, inputs))
            print(self.task.name)
            for input in self.inputs:
                print(input)
            print(self.outputs[0])
        elif self.current_mode == self.lazy_mode:
            output = Tensor.from_type(ttype=task.type_of_param(task.compute), op=self, index=0)
            self.outputs.append(output)
        else:
            raise NotImplementedError('coming soon')

    @classmethod
    def set_execution_mode(cls, mode):
        assert mode in [cls.imperative_mode,
                        cls.imperative_opt_mode,
                        cls.lazy_mode]
        cls.current_mode = mode


def execution_mode(mode):
    Operator.set_execution_mode(mode)


def convert(v):
    if isinstance(v, float):
        return Tensor(shape=[1], dtype='float32', name='scalar_const', value=TensorValue.full(shape=[1], scalar_type='float32', scope='global', fill_value=v))
    elif isinstance(v, Tensor):
        return v
    else:
        raise NotImplementedError()


class Tensor:
    def __init__(self, shape, dtype: str, layout=None, producer=None, index=0, name=None, value=None, init_method=None):
        self.dtype: ScalarType = ScalarType(dtype) if isinstance(dtype, str) else dtype
        self.shape: List[int] = [int(v) for v in shape]
        self.layout: DataLayout = layout if layout else DataLayout.row_major(shape)
        self.name: str = name
        self.producer: Optional[Operator] = None
        self.index: Optional[int] = None
        if Operator.current_mode == Operator.lazy_mode:
            self.producer = producer
            self.index = index
        self.attached_value: Optional[TensorValue] = value
        if value is None and init_method is not None:
            assert init_method == 'rand'
            self.attached_value = randn(shape=shape, scalar_type='float32', scope='global')

    @staticmethod
    def from_type(ttype: TensorType, op, index) -> 'Tensor':
        return Tensor(shape=ttype.shape, dtype=ttype.scalar_type.name, layout=ttype.layout, producer=op, index=index)

    def value(self) -> TensorValue:
        if self.attached_value is not None:
            return self.attached_value
        else:
            assert Operator.current_mode == Operator.lazy_mode
            raise NotImplementedError()

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

    def __str__(self):
        if self.attached_value is None:
            return 'Tensor(shape={}, dtype={}, attach_value=None)'.format(self.shape, self.dtype.name)
        else:
            return 'Tensor(shape={}, dtype={}): \n{}'.format(self.shape, self.dtype.name, self.attached_value)

    def reshape(self, shape):
        from .ops import reshape
        return reshape(self, shape)

    def flatten(self, start_dim=0, end_dim=-1):
        from .ops import flatten
        return flatten(self, start_dim, end_dim)

    def rsqrt(self):
        from .ops import rsqrt
        return rsqrt(self)


class Module:
    def __init__(self):
        self.name = None
        self.parameters: OrderedDict[str, Optional[Tensor]] = OrderedDict()
        self.submodules: OrderedDict[str, Optional[Module]] = OrderedDict()

    def __setattr__(self, key, value):
        parameters = self.__dict__.get('parameters')
        submodules = self.__dict__.get('submodules')
        if isinstance(value, Tensor):
            value.name = key
            self.parameters[key] = value
        elif isinstance(value, Module):
            value.name = '{}.{}'.format(self.name, key) if self.name else key
            self.submodules[key] = value
        elif parameters and submodules and value is None and (key in parameters or key in submodules):
            if key in self.parameters:
                self.parameters[key] = value
            if key in self.submodules:
                self.submodules[key] = value
        else:
            super().__setattr__(key, value)
        cnt = sum([1 for collection in [parameters, submodules, self.__dict__] if collection and key in collection])
        assert cnt <= 1, 'duplicated definition of {}'.format(key)

    def __getattr__(self, item):
        if item in self.parameters:
            return self.parameters[item]
        if item in self.submodules:
            return self.submodules[item]
        raise AttributeError(item)

    def __str__(self):
        lines = []
        args_lines = self.extra_str().split('\n')
        lines.extend([line for line in args_lines if len(line) > 0])
        for key, submodule in self.submodules.items():
            substr = str(submodule)
            sub_lines = substr.split('\n')
            sub_lines[0] = '({}): {}'.format(key, sub_lines[0])
            lines.extend(sub_lines)
        indent = 2
        name = self.__class__.__name__
        if len(lines) <= 1:
            return '{}({})'.format(name, '\n'.join(lines))
        else:
            lines = [' ' * indent + line for line in lines]
            return '{}(\n{}\n)'.format(name, '\n'.join(lines))

    def __call__(self, *args):
        return self.forward(*args)

    def extra_str(self) -> str:
        return ''

    def forward(self, *args):
        raise NotImplementedError()
