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
from hidet.ir.type import DataType
from .integer import IntegerType, IntInfo, uint8, uint32


class IntegerSubbyteType(IntegerType):
    def __init__(self, name, short_name, storage, nbits, signed, min_value, max_value):
        nbytes = storage.nbytes
        super().__init__(name, short_name, nbytes, min_value, max_value)
        self._storage: DataType = storage
        self._nbits: int = nbits
        self._signed: bool = signed
        self._bits_mask: int = (1 << self._nbits) - 1
        self._sign_mask: int = 1 << (self._nbits - 1) if self._signed else 0

    def iinfo(self) -> IntInfo:
        return IntInfo(self._nbits, self._max_value, self._min_value, self)


int4b = IntegerSubbyteType('int4b', 'i4', uint8, 4, True, -8, 7)
int3b = IntegerSubbyteType('int3b', 'i3', uint32, 3, True, -4, 3)
int2b = IntegerSubbyteType('int2b', 'i2', uint8, 2, True, -2, 1)
int1b = IntegerSubbyteType('int1b', 'i1', uint8, 1, True, -1, 0)

uint4b = IntegerSubbyteType('uint4b', 'u4', uint8, 4, False, 0, 16)
uint3b = IntegerSubbyteType('uint3b', 'u3', uint32, 3, False, 0, 8)
uint2b = IntegerSubbyteType('uint2b', 'u2', uint8, 2, False, 0, 4)
uint1b = IntegerSubbyteType('uint1b', 'u1', uint8, 1, False, 0, 1)

i4 = int4b
i3 = int3b
i2 = int2b
i1 = int1b

u4 = uint4b
u3 = uint3b
u2 = uint2b
u1 = uint1b
