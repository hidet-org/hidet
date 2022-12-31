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
from typing import Any
from dataclasses import dataclass
import warnings
from hidet.ir.type import DataType


@dataclass
class IntInfo:
    bits: int
    max: int
    min: int
    dtype: DataType


class IntegerType(DataType):
    def __init__(self, name, short_name, nbytes, min_value, max_value):
        super().__init__(name, short_name, nbytes)
        self._min_value: int = min_value
        self._max_value: int = max_value

    def is_float(self) -> bool:
        return False

    def is_integer(self) -> bool:
        return True

    def is_vector(self) -> bool:
        return False

    def constant(self, value: Any):
        from hidet.ir.expr import Constant

        if isinstance(value, Constant):
            value = value.value
        if isinstance(value, float):
            warnings.warn('Converting float to integer when creating {} constant: {}.'.format(self.name, value))
        value = int(value)

        if not self._min_value <= value <= self._max_value:
            raise ValueError('Value {} is out of range for {}.'.format(value, self.name))
        return Constant(value, self)

    @property
    def one(self):
        return self.constant(1)

    @property
    def zero(self):
        return self.constant(0)

    @property
    def min_value(self):
        return self.constant(self._min_value)

    @property
    def max_value(self):
        return self.constant(self._max_value)

    def iinfo(self) -> IntInfo:
        return IntInfo(self.nbytes * 8, self._max_value, self._min_value, self)


int8 = IntegerType('int8', 'i8', 1, -128, 127)
int16 = IntegerType('int16', 'i16', 2, -32768, 32767)
int32 = IntegerType('int32', 'i32', 4, -2147483648, 2147483647)
int64 = IntegerType('int64', 'i64', 8, -9223372036854775808, 9223372036854775807)

uint8 = IntegerType('uint8', 'u8', 1, 0, 255)
uint16 = IntegerType('uint16', 'u16', 2, 0, 65535)
uint32 = IntegerType('uint32', 'u32', 4, 0, 4294967295)
uint64 = IntegerType('uint64', 'u64', 8, 0, 18446744073709551615)

i8 = int8
i16 = int16
i32 = int32
i64 = int64

u8 = uint8
u16 = uint16
u32 = uint32
u64 = uint64
