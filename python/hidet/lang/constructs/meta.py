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
