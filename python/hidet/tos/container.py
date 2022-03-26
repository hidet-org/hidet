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
