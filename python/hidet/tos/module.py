from typing import Optional
from collections import OrderedDict
from hidet.tos.tensor import Tensor


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
