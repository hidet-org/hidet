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
from typing import Optional, Sequence
from collections import OrderedDict
from hidet.graph.tensor import symbol_like
from hidet.graph.ir.flow_graph import FlowGraph, trace_from
from hidet.graph.tensor import Tensor


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
        cnt = sum(1 for collection in [parameters, submodules, self.__dict__] if collection and key in collection)
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

    def flow_graph_for(self, inputs: Sequence[Tensor]) -> FlowGraph:
        symbol_inputs = []
        for arg in inputs:
            if isinstance(arg, Tensor):
                symbol_inputs.append(symbol_like(arg))
            else:
                raise ValueError('Currently only support Tensor as input when automatically creating flow_graph.')
        symbol_outputs = self.forward(*symbol_inputs)
        return trace_from(symbol_outputs, symbol_inputs)

    def to_cuda(self) -> Module:
        for name, submodule in self.submodules.items():
            submodule.to_cuda()
        for name, parameter in self.parameters.items():
            self.parameters[name] = parameter.cuda()
        return self
