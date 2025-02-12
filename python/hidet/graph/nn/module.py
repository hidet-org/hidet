# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations
from typing import Optional, Sequence, Iterator, Dict, Any, Generic, TypeVar
from collections import OrderedDict
from hidet.graph.tensor import symbol_like
from hidet.graph.flow_graph import FlowGraph, trace_from
from hidet.graph.tensor import Tensor

# forward method return type
R = TypeVar('R')


class Module(Generic[R]):
    def __init__(self):
        self.name = None
        self._parameters: OrderedDict[str, Optional[Tensor]] = OrderedDict()
        self._submodules: OrderedDict[str, Optional[Module]] = OrderedDict()

    def __setattr__(self, key, value):
        if key in ['name', '_submodules', '_parameters']:
            super().__setattr__(key, value)
            return

        parameters = self.__dict__.get('_parameters')
        submodules = self.__dict__.get('_submodules')

        if key in parameters:
            del self._parameters[key]
        elif key in submodules:
            del self._submodules[key]
        elif key in self.__dict__:
            del self.__dict__[key]

        if isinstance(value, Tensor):
            parameters[key] = value
        elif isinstance(value, Module):
            submodules[key] = value
        else:
            self.__dict__[key] = value

        cnt = sum(1 for collection in [parameters, submodules, self.__dict__] if collection and key in collection)
        assert cnt <= 1, 'duplicated definition of {}'.format(key)

    def __getattr__(self, item):
        if item == '_parameters':
            return super().__getattribute__(item)
        if item == '_submodules':
            return super().__getattribute__(item)
        if item in self._parameters:
            return self._parameters[item]
        if item in self._submodules:
            return self._submodules[item]
        raise AttributeError(item)

    def __str__(self):
        lines = []
        args_lines = self.extra_str().split('\n')
        lines.extend([line for line in args_lines if len(line) > 0])
        for key, submodule in self._submodules.items():
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

    def __call__(self, *args, **kwargs) -> R:
        return self.forward(*args, **kwargs)

    def state_dict(self) -> Dict[str, Any]:
        state_dict = OrderedDict()
        for name, parameter in self.named_parameters():
            state_dict[name] = parameter
        return state_dict

    def load_state_dict(self, state_dict: Dict[str, Any]):
        for name, parameter in self.named_parameters():
            parameter.copy_(state_dict[name])

    def extra_str(self) -> str:
        return ''

    def forward(self, *args, **kwargs) -> R:
        raise NotImplementedError()

    def parameters(self, recursive: bool = True) -> Iterator[Tensor]:
        for _, parameter in self.named_parameters(recursive=recursive):
            yield parameter

    def named_parameters(self, prefix='', recursive=True):
        for name, parameter in self._parameters.items():
            yield name, parameter
        if recursive:
            for module_name, submodule in self._submodules.items():
                for name, parameter in submodule.named_parameters(prefix, recursive):
                    param_name = '{}{}.{}'.format(prefix + '.' if prefix else '', module_name, name)
                    yield param_name, parameter

    def flow_graph_for(self, inputs: Sequence[Tensor]) -> FlowGraph:
        symbol_inputs = []
        for arg in inputs:
            if isinstance(arg, Tensor):
                symbol_inputs.append(symbol_like(arg))
            else:
                raise ValueError('Currently only support Tensor as input when automatically creating flow_graph.')
        symbol_outputs = self.forward(*symbol_inputs)
        return trace_from(symbol_outputs, symbol_inputs)

    def cpu(self) -> Module:
        for name, submodule in self._submodules.items():
            submodule.cpu()
        for name, parameter in self._parameters.items():
            self._parameters[name] = parameter.cpu()
        return self

    def cuda(self) -> Module:
        for name, submodule in self._submodules.items():
            submodule.cuda()
        for name, parameter in self._parameters.items():
            self._parameters[name] = parameter.cuda()
        return self

    def hip(self) -> Module:
        for name, submodule in self._submodules.items():
            submodule.hip()
        for name, parameter in self._parameters.items():
            self._parameters[name] = parameter.hip()
        return self

    def to(self, dtype=None, device=None) -> Module:
        for name, submodule in self._submodules.items():
            submodule.to(dtype, device)
        for name, parameter in self._parameters.items():
            self._parameters[name] = parameter.to(dtype, device)
        return self
