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
from typing import Iterable
import numpy as np


class HashSum:
    def __init__(self, obj):
        if isinstance(obj, np.ndarray):
            self.value = id(obj)
        else:
            self.value = hash(obj)
        self.hashed_obj = obj

    def __str__(self):
        return str(self.value % 107)

    def __add__(self, other):
        return HashSum((self.value, other))

    def __iadd__(self, other):
        self.value = HashSum((self.value, other.value)).value
        return self

    def __and__(self, other):
        return HashSum.hash_set([self, other])

    def __hash__(self):
        return self.value

    def __eq__(self, other):
        assert isinstance(other, HashSum)
        return self.value == other.value

    @staticmethod
    def hash_set(objs: Iterable) -> 'HashSum':
        return HashSum(tuple(sorted([hash(obj) for obj in objs])))
