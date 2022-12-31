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
from typing import Iterable
from collections import OrderedDict
from hidet.graph.module import Module


class Sequential(Module):
    def __init__(self, *args):
        super().__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.__setattr__(key, module)
        elif len(args) == 1 and isinstance(args[0], (tuple, list)):
            for i, module in enumerate(args[0]):
                self.__setattr__(str(i), module)
        else:
            for idx, module in enumerate(args):
                self.__setattr__(str(idx), module)

    def forward(self, x):  # pylint: disable=arguments-differ
        for module in self.submodules.values():
            x = module(x)
        return x


class ModuleList(Module):
    def __init__(self, modules: Iterable[Module] = None):
        super().__init__()
        for idx, module in enumerate(modules):
            self.submodules[str(idx)] = module

    def forward(self, *args):
        raise ValueError('Should not forward ModuleList.')
