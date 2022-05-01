from __future__ import annotations
from typing import Optional, Iterable
from collections import OrderedDict
from hidet.tos.module import Module


class Sequential(Module):
    def __init__(self, *args):
        super().__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.__setattr__(key, module)
        else:
            for idx, module in enumerate(args):
                self.__setattr__(str(idx), module)

    def forward(self, x):
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
