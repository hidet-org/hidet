from typing import Sequence, List, Type, Union
import builtins
from hidet.ir.type import BaseType


class HidetMetaLoopIterable:
    def __init__(self, iterable):
        self.iterable = iterable

    def __iter__(self):
        return iter(self.iterable)


class HidetMetaParamTypeList:
    def __init__(self, arg_types: Sequence[BaseType]):
        self.arg_types: List[BaseType] = list(arg_types)


def range(extent: int):
    return HidetMetaLoopIterable(builtins.range(extent))


def each(iterable):
    return HidetMetaLoopIterable(iterable)


def types(arg_types: Sequence[Union[BaseType, Type[Union[int, float, bool]]]]):
    return HidetMetaParamTypeList(arg_types)
